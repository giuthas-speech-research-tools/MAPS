import os

from maps.cli import run_maps_cli

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from textgrid import textgrid
from tensorflow.keras.models import load_model
from scipy.io import wavfile
import numpy as np
import re, sys, itertools
from tqdm import tqdm
import tensorflow as tf
import python_speech_features as psf
from utils import align, collapse, load_dictionary
from pathlib import Path
from args import build_arg_parser
import statistics
import math
import warnings
import natsort
import soxr
import tempfile

EPS = 1e-8

FRAME_LENGTH = 0.025 # 25 ms expressed as seconds
FRAME_INTERVAL = 0.01 # 10 ms expressed as seconds

phones = 'h#	q	eh	dx	iy	r	ey	ix	tcl	sh	ow	z	s	hh	aw	m	t	er	l	w	aa	hv	ae	dcl	y	axr	d	kcl	k	ux	ng	gcl	g	ao	epi	ih	p	ay	v	n	f	jh	ax	en	oy	dh	pcl	ah	bcl	el	zh	uw	pau	b	uh	th	ax-h	em	ch	nx	eng'.split()

num2phn = {i: p for i, p in enumerate(phones)}
phn2num = {p: i for i, p in enumerate(phones)}
phn2num['sil'] = phn2num['h#']

class PhoneLabel:

    def __init__(self, phone, duration):
    
        self.phone = phone
        self.duration = duration
        
    def __str__(self):
    
        return str([self.phone, self.duration])
        
class WordString:

    def __init__(self, words, pronunciations):
        self.words = words
        self.pronunciations = pronunciations
        
        self.phone_string = list(itertools.chain(*pronunciations))
        self.collapsed_string = collapse([re.sub(r'[0-9]', '', x) for x in self.phone_string])
        
        self.did_collapse = len(self.phone_string) != len(self.collapsed_string)
        
    def __str__(self):
        return str([self.words, f'collapsed_diff={self.did_collapse}', self.pronunciations])
    
def force_align(collapsed, yhat):

    yhat = np.squeeze(yhat, 0)
    predictions = np.abs(np.log(yhat))
    collapsed = [phn2num[x.lower()] for x in collapsed]
    a, M = align(collapsed, predictions)
    a = [num2phn[p] for p in a]
    seq = [PhoneLabel(phone=a[0], duration=1)]
    durI = 1
    for elem in a[1:]:
        if not seq[-1].phone == elem:
            pl = PhoneLabel(phone=elem, duration=1)
            seq.append(pl)
        else:
            seq[-1].duration += 1
    return seq, M
    
def make_textgrid(seq, tgname, maxTime, words, interpolate=True, probs=None):
    '''
    Side-effect of writing TextGrid to disk
    '''
    
    if interpolate and np.all(probs == None):
    
        raise ValueError('If using interpolation, the alignment matrix must also be passed in through the probs argument')
        
    
    tg = textgrid.TextGrid()
    tier = textgrid.IntervalTier()
    tier.name = 'phones'
    curr_dur = 0
    
    if len(seq) == 1:
        last_interval = textgrid.Interval(curr_dur, maxTime, seq[-1].phone)
        tier.intervals.append(last_interval)
        if words.did_collapse: unmerge_phones(tier, words)
        word_tier = make_word_tier(tier, words)
        tg.tiers.append(word_tier)
        tg.tiers.append(tier)
        tg.write(tgname)
        return
    
    added_bits = []
    frame_durs = [s.duration for s in seq]
    cumu_frame_durs = [sum(frame_durs[0:i+1]) for i in range(len(frame_durs))]
    curr_dur = seq[0].duration * FRAME_INTERVAL + 0.015
    
    if interpolate:
    
        additional = interpolated_part(seq[0].duration-1, 0, probs)
        if curr_dur + additional < maxTime:
            curr_dur += additional
        added_bits.append(additional)
    
    tier.intervals.append(textgrid.Interval(0, curr_dur, seq[0].phone))

    for i, s in enumerate(seq[:-1]):
    
        if i == 0: continue
    
        label = s.phone
        duration = s.duration
    
        beginning = curr_dur
        dur = FRAME_INTERVAL * duration
        
        if interpolate:
        
            endCur = cumu_frame_durs[i] - 1
        
            dur -= added_bits[-1]
            additional = interpolated_part(endCur, i, probs)
            if beginning + dur + additional < maxTime:
                dur += additional
            added_bits.append(additional)
        
        ending = beginning + dur
        
        interval = textgrid.Interval(beginning, ending, label)
        tier.intervals.append(interval)

        curr_dur = ending
    
    last_interval = textgrid.Interval(curr_dur, maxTime, seq[-1].phone)
    tier.intervals.append(last_interval)
    
    if words.did_collapse:
        unmerge_phones(tier, words)

    for i in range(len(tier.intervals)):
        x = words.phone_string[i]
        y = tier.intervals[i]
        if x != y.mark:
            tier.intervals[i].mark = x

        # Prevent small boundary errors from numerical instability
        if i > 0:
            prev_end = tier.intervals[i-1].maxTime
            curr_start = tier.intervals[i].minTime

            if math.isclose(prev_end, curr_start) and not prev_end == curr_start:
                tier.intervals[i-1].maxTime = curr_start

    word_tier = make_word_tier(tier, words)
    tg.tiers.append(word_tier)
    tg.tiers.append(tier)
    
    tg.write(tgname)

def to_bucket_fmt(s):

    s = [re.sub(r'[0-9]', '', x) for x in s]

    buckets = []
    prev = s[0]
    count = 1
    for x in s[1:]:
        if x != prev:
            buckets.append((prev, count))
            count = 0
        prev = x
        count += 1
    buckets.append((prev, count))
    return buckets
    
def unmerge_phones(tier, words):

    collapsed_bucket = to_bucket_fmt(words.collapsed_string)
    uncollapsed_bucket = to_bucket_fmt(words.phone_string)

    intervals = []

    for i, (c, u) in enumerate(zip(collapsed_bucket, uncollapsed_bucket)):

        dur = tier.intervals[i].maxTime - tier.intervals[i].minTime
        chunk_dur = dur / u[1]
        mint = tier.intervals[i].minTime

        for j in range(u[1]):
            low = mint + j * chunk_dur
            high = low + chunk_dur
            interv = textgrid.Interval(minTime=low, maxTime=high, mark=c[0])
            intervals.append(interv)

    tier.intervals = intervals
    return
    
def make_word_tier(segment_tier, words):

    phone_string = list(itertools.chain(words.phone_string))
    
    words_int = textgrid.IntervalTier()
    words_int.name = 'words'
    word_ends = np.cumsum([len(p) for p in words.pronunciations]) - 1
    maxTime = segment_tier[word_ends[0]].maxTime
    interval = textgrid.Interval(minTime=0, maxTime=maxTime, mark=words.words[0])
    words_int.intervals.append(interval)
    
    for w, w_end in zip(words.words[1:], word_ends[1:]):
        minTime = words_int[-1].maxTime
        maxTime = segment_tier[w_end].maxTime
        interval = textgrid.Interval(minTime=minTime, maxTime=maxTime, mark=w)
        words_int.intervals.append(interval)
    
    return words_int
    
def interpolated_part(endCur, phone_n, probs):

    phone1_curr = probs[endCur, phone_n]
    phone1_next = probs[endCur+1, phone_n]

    phone2_curr = probs[endCur, phone_n+1]
    phone2_next = probs[endCur+1, phone_n+1]
        
    m1 = (phone1_next - phone1_curr) / FRAME_INTERVAL
    m2 = (phone2_next - phone2_curr) / FRAME_INTERVAL
    
    A = np.array([[-m1, 1], [-m2, 1]])
    b = [phone1_curr, phone2_curr]
    
    try:
        time_point, intersection_probability = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return 0
    
    curr_dur = seq[0].duration * FRAME_INTERVAL + 0.015
    if 0 <= time_point < FRAME_INTERVAL:
        return time_point
        
    return 0


if __name__ == '__main__':

    run_maps_cli()

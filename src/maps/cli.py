from args import build_arg_parser
from maps.maps_api import EPS, force_align, FRAME_INTERVAL, make_textgrid, \
    make_word_tier, WordString
from maps.utils import load_dictionary


def run_maps_cli():
    global args, resample, duration, mfcc, delta, seq, intervals, tier
    p = build_arg_parser()
    args = p.parse_args()
    args = vars(args)
    wavname_path = Path(args['audio'])
    if not wavname_path.is_file() and not wavname_path.is_dir():
        raise RuntimeError(
            f'Could not find {wavname_path}. Please check the spelling and try again.')
    elif wavname_path.is_dir():
        wavnames = [wavname_path / Path(x) for x in os.listdir(wavname_path) if
                    x.lower().endswith('.wav')]
        wavnames.sort()
    else:
        wavnames = [wavname_path]
    resample = args['resample']
    if resample:
        resampled_wavnames = list()
        print('RESAMPLING TO 16,000 HZ...')
        temp_d = Path(tempfile.mkdtemp())
        for wname in tqdm(wavnames):
            sr, samples = wavfile.read(wname)
            samples = soxr.resample(samples, sr, 16_000)
            out_path = temp_d / wname.name
            wavfile.write(out_path, 16_000, samples)
            resampled_wavnames.append(out_path)
        wavnames = resampled_wavnames
    model_path = Path(args['model'])
    if not model_path.suffix == '.tf':
        model_names = natsort.natsorted(
            [x for x in model_path.iterdir() if x.suffix == '.tf'])
        if not model_names:
            raise RuntimeError(
                f'Could not find a model named {model_path}, nor any models within that path. Please check spelling and file extensions and try again.')
    else:
        model_names = [model_path]
    use_ensemble = len(model_names) > 1
    rm_ensemble = args['rm_ensemble']
    ensemble_table = args['ensemble_table']
    transcription_path = Path(args['text'])
    if not transcription_path.is_file() and not transcription_path.is_dir():
        raise RuntimeError(
            f'Could not find {transcription_path}. Please check the spelling and try again.')
    elif transcription_path.is_dir():
        transcriptions = [transcription_path / Path(x.name).with_suffix('.txt')
                          for x in wavnames]
    else:
        transcriptions = [transcription_path]
    w_set = set(x.stem for x in wavnames)
    t_set = set(x.stem for x in transcriptions)
    mismatched = []
    for w in wavnames:
        if w.stem not in t_set:
            mismatched.append(w)
    for t in transcriptions:
        if t.stem not in w_set:
            mismatched.append(t)
    if mismatched:
        raise RuntimeError(
            f'The following files did not have a corresponding WAV or txt match. Please add matches or remove the files. Note that name matching is case-sensitive.\n{",".join(str(x) for x in mismatched)}')
    d_path = Path(args['dict'])
    if not d_path.is_file():
        raise RuntimeError(
            f'Could not find {d_path}. Please check the spelling and try again.')
    word2phone = load_dictionary(d_path)
    # Parentheses to help visually distinguish the "=" and "==" in the same line
    use_interp = (args['interp'] == 'true')
    add_sil = (args['sil'] == 'true')
    tgnames = [x.with_suffix('.TextGrid') for x in wavnames]
    word_list = []
    for t in transcriptions:
        with open(t, 'r') as f:
            w = f.read().upper().split()
            word_list += w
    ood_words = set([w for w in word_list if w not in word2phone])
    if ood_words:
        raise RuntimeError(
            f'The following words were not found in the dictionary. Please add them to the dictionary and run the aligner again.\n{", ".join(ood_words)}')
    quiet = args['quiet']
    filenames = list(zip(tgnames, wavnames, transcriptions))
    if not quiet:
        print('BEGINNING ALIGNMENT')
    overwrite = args['overwrite']
    for m_I, m_name in enumerate(model_names, start=1):

        # if m_name.suffix == '.tf':
        #     warnings.warn('TensorFlow has stopped supporting the tf format. Your models may need to be updated to the keras or h5 formats for long-term functionality.')
        #     m = tf.keras.layers.TFSMLayer(m_name, call_endpoint='serving_default')
        m = load_model(m_name, compile=False)

        print(f'USING MODEL {m_name.name} ({m_I}/{len(model_names)})',
              flush=True)

        if not quiet:
            filenames = tqdm(filenames)

        for tgname_base, wavname, transcription in filenames:

            if use_ensemble:
                tgname = tgname_base.parent / tgname_base.parts[-1].replace(
                    '.TextGrid', f'_{m_name.stem}.TextGrid')
            else:
                tgname = tgname_base

            if tgname.is_file() and not overwrite:
                continue

            sr, samples = wavfile.read(wavname)
            duration = samples.size / sr  # convert samples to seconds

            mfcc = psf.mfcc(samples, sr, winstep=FRAME_INTERVAL)
            delta = psf.delta(mfcc, 2)
            deltadelta = psf.delta(delta, 2)

            x = np.hstack((mfcc, delta, deltadelta))
            x = np.expand_dims(x, axis=0)

            yhat = m.predict(x, verbose=0)

            with open(transcription, 'r') as f:
                word_labels = f.read().upper().split()

            if add_sil and duration >= 0.045:
                word_labels = ['sil'] + word_labels + ['sil']
            elif add_sil:
                warnings.warn(
                    f'Silence segments not added to ends of transcription for {wavname} because duration of {duration} s is too short to have silence padding.')
            word_chain = [word2phone[w] for w in word_labels]

            best_score = np.inf

            check_variants = args['check_variants']
            best_w_string = 0

            # Iterate through pronunciation variants to choose best alignment
            # TODO: This iteration only checks segmental differences; stress differences won't get evaluated
            #   and may end up semi-randomly chosen (or choose only first option)
            #
            # This method will very quickly cause combinatoric explosion since function words have
            # several variants
            for c in itertools.product(*word_chain):

                # Remove empty 'sil' options
                if add_sil:
                    this_word_labels = [x for cI, x in zip(c, word_labels) if
                                        cI]
                    c = [x for x in c if x]
                else:
                    this_word_labels = word_labels

                w_string = WordString(this_word_labels, c)
                if add_sil and len(w_string.collapsed_string) > (
                        duration - 0.015) / 0.01:
                    if this_word_labels[0] == 'sil':
                        this_word_labels = this_word_labels[1:]
                        c = c[1:]
                        warnings.warn(
                            f'File {wavname} with duration {duration} too short for adding silence to transcription {w_string.collapsed_string}. Removing first silence label.')
                    if this_word_labels[-1] == 'sil':
                        this_word_labels = this_word_labels[:-1]
                        c = c[:-1]
                        warnings.warn(
                            f'File {wavname} with duration {duration} too short for adding silence to transcription {w_string.collapsed_string}. Removing final silence label.')

                    w_string = WordString(this_word_labels, c)

                if best_w_string == 0:
                    best_w_string = w_string

                seq, M = force_align(w_string.collapsed_string, yhat)
                if M[-1, -1] < best_score:
                    best_seq = seq
                    best_M = M
                    best_score = M[-1, -1]
                    best_w_string = w_string

                if not check_variants:
                    break

            n_segs = len(best_w_string.collapsed_string)
            if n_segs > 1 and duration < 0.015 + (0.01 * n_segs):
                warnings.warn(
                    f'File {wavname} with duration {duration} too short for collapsed {len(best_w_string.collapsed_string)}-segment best transcription {best_w_string.collapsed_string}. Assigning equal durations for each segment.')

                intervals = []
                for i, x in enumerate(best_w_string.collapsed_string):
                    d_min = i / len(best_w_string.collapsed_string) * duration
                    d_max = (i + 1) / len(
                        best_w_string.collapsed_string) * duration

                    intervals.append(
                        textgrid.Interval(minTime=d_min, maxTime=d_max, mark=x))

                tier = textgrid.IntervalTier('segments')
                tier.intervals = intervals
                word_tier = make_word_tier(tier, best_w_string)
                tg = textgrid.TextGrid()
                tg.tiers.append(word_tier)
                tg.tiers.append(tier)
                tg.write(tgname)
                continue

            make_textgrid(best_seq, tgname, duration, best_w_string,
                          interpolate=use_interp, probs=best_M.T)
    if use_ensemble:
        print('ENSEMBLING', flush=True)

        if ensemble_table:
            f_path = f'{"_".join(wavname_path.parts)}_{model_path.name}_alignment_results.tsv'
            col_names = ['file', 'word', 'word_mintime', 'word_maxtime',
                         'segment', 'segment_mintime', 'segment_maxtime',
                         'segment_lo_ci', 'segment_hi_ci']
            with open(f_path, 'a') as w:
                w.write('\t'.join(col_names) + '\n')

        all_tg_names = list()
        for tgname_base, _, _ in tqdm(filenames):

            ensemble_tg_path = Path(tgname_base.parent,
                                    tgname_base.stem + '_ensemble.TextGrid')

            ens_intervals = textgrid.IntervalTier(name='segments')
            intervals = list()
            cis = list()

            tg_names = list()

            for m_name in model_names:
                tail = tgname_base.parts[-1].replace('.TextGrid',
                                                     f'_{m_name.stem}.TextGrid')
                t = tgname_base.parent / tail
                tg_names.append(t)

            all_tg_names += tg_names
            if ensemble_tg_path.is_file() and not overwrite:
                continue

            tgs = [textgrid.TextGrid() for _ in tg_names]
            for tg, tg_name in zip(tgs, tg_names):
                tg.read(tg_name, round_digits=1000)

            n_tgs = len(tgs)

            n_intervals = len(tgs[0].tiers[1].intervals)
            duration = tgs[0].tiers[1].maxTime

            for i in range(n_intervals):
                lab = tgs[0].tiers[1].intervals[i].mark
                mintimes = [tgs[tier_I].tiers[1].intervals[i].minTime for tier_I
                            in range(n_tgs)]
                maxtimes = [tgs[tier_I].tiers[1].intervals[i].maxTime for tier_I
                            in range(n_tgs)]

                mintime = statistics.median(mintimes)
                maxtime = statistics.median(maxtimes)

                times_sorted = sorted(maxtimes)
                ci_lo = times_sorted[1]
                ci_hi = times_sorted[8]
                if ci_lo == ci_hi:
                    ci_lo -= EPS
                    ci_hi += EPS

                interval = textgrid.Interval(minTime=mintime, maxTime=maxtime,
                                             mark=lab)
                intervals.append(interval)

                if i < n_intervals - 1:
                    ci_lo_p = textgrid.Point(mark=f'{lab}_cilo', time=ci_lo)
                    ci_hi_p = textgrid.Point(mark=f'{lab}_cihi', time=ci_hi)
                    cis += [ci_lo_p, ci_hi_p]

            n_word_intervals = len(tgs[0].tiers[0].intervals)
            word_intervals = list()

            for i in range(n_word_intervals):
                lab = tgs[0].tiers[0].intervals[i].mark
                mintimes = [tgs[tier_I].tiers[0].intervals[i].minTime for tier_I
                            in range(n_tgs)]
                maxtimes = [tgs[tier_I].tiers[0].intervals[i].maxTime for tier_I
                            in range(n_tgs)]

                mintime = statistics.median(mintimes)
                maxtime = statistics.median(maxtimes)

                word_interval = textgrid.Interval(minTime=mintime,
                                                  maxTime=maxtime, mark=lab)
                word_intervals.append(word_interval)

            # make word tier here

            ens_tg = textgrid.TextGrid(maxTime=tgs[0].maxTime)

            word_tier = textgrid.IntervalTier(name='words')
            word_tier.intervals = word_intervals
            ens_tg.tiers.append(word_tier)

            int_tier = textgrid.IntervalTier(name='segments')
            int_tier.intervals = intervals
            ens_tg.tiers.append(int_tier)

            ci_tier = textgrid.PointTier(name='95-CIs')
            ci_tier.points = cis
            ens_tg.tiers.append(ci_tier)

            ens_tg.write(ensemble_tg_path)

            all_tg_names += tg_names

            if ensemble_table:
                with open(f_path, 'a') as w:

                    fname = ensemble_tg_path.name

                    word_iter = iter(word_intervals)
                    word = next(word_iter)

                    for x_I, x in enumerate(intervals):
                        if x_I == len(intervals) - 1:
                            segment_lo_ci = x.maxTime
                            segment_hi_ci = x.maxTime
                        else:
                            segment_lo_ci = cis[x_I * 2].time
                            segment_hi_ci = cis[x_I * 2 + 1].time

                        s = [fname, word.mark, word.minTime, word.maxTime,
                             x.mark, x.minTime, x.maxTime, segment_lo_ci,
                             segment_hi_ci]
                        s = '\t'.join([str(z) for z in s])

                        w.write(s + '\n')

                        if x.maxTime == word.maxTime and x_I < len(
                                intervals) - 1:
                            word = next(word_iter)

        # Remove ensemble files if flagged to remove
        if rm_ensemble:
            for n in all_tg_names:
                n.unlink(missing_ok=True)

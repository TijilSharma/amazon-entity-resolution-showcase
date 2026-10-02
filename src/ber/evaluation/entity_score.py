"""Challenge macro F0.5 from the organizer README, including singleton credit."""
import math


def entity_f05(tp, fp, fn):
    denominator = 1.25 * tp + fp + .25 * fn
    return 1.25 * tp / denominator if denominator else 1.0


def score(truth, predictions):
    if not truth:
        raise ValueError('Cannot score an empty anchor population')
    if set(predictions) - set(truth):
        raise ValueError('Predictions contain unknown anchors')
    details = []
    for anchor, values in truth.items():
        correct = set(values)
        listed = predictions.get(anchor, [])
        if len(listed) != len(set(listed)):
            raise ValueError('Duplicate predicted target')
        predicted = set(listed)
        tp, fp, fn = len(correct & predicted), len(predicted-correct), len(correct-predicted)
        details.append({'source1_entity_id': anchor, 'tp':tp, 'fp':fp, 'fn':fn,
                        'singleton':not bool(correct), 'f05':entity_f05(tp,fp,fn)})
    singleton = [r for r in details if r['singleton']]
    return {'macro_f05':sum(r['f05'] for r in details)/len(details), 'anchors':len(details),
            'tp':sum(r['tp'] for r in details), 'fp':sum(r['fp'] for r in details),
            'fn':sum(r['fn'] for r in details), 'singleton_anchors':len(singleton),
            'singleton_accuracy':sum(r['f05'] for r in singleton)/len(singleton) if singleton else None,
            'per_anchor':details}


def decisions(anchors, pairs, probabilities, threshold):
    if not math.isfinite(threshold):
        raise ValueError('Nonfinite threshold')
    if len(pairs) != len(probabilities):
        raise ValueError('Pair/score lengths differ')
    predictions={q:[] for q in anchors}
    seen=set()
    for (q,t), probability in zip(pairs,probabilities):
        if q not in predictions or (q,t) in seen:
            raise ValueError('Unknown anchor or duplicate pair')
        seen.add((q,t))
        if not math.isfinite(float(probability)) or not 0 <= probability <= 1:
            raise ValueError('Invalid probability')
        if probability >= threshold:
            predictions[q].append(t)
    return predictions


def choose_threshold(truth, pairs, probabilities):
    """Exact calibration sweep, including reject-all and equal-score batches.

    Prefer the higher threshold when macro F0.5 ties. Never choose a threshold
    by scoring the holdout set. Unretrieved positives remain false negatives.
    """
    decisions(truth,pairs,probabilities,2.0)  # validate alignment, duplicates and scores
    if not truth:
        raise ValueError('Empty calibration truth')
    correct={q:set(v) for q,v in truth.items()}
    state={q:[0,0,len(v)] for q,v in correct.items()}
    total=sum(entity_f05(*v) for v in state.values())
    threshold=math.nextafter(max(map(float,probabilities),default=1.0), math.inf)
    best={'threshold':threshold,'macro_f05':total/len(truth)}
    curve=[dict(best)]
    events=sorted(zip(probabilities,pairs),key=lambda item:-float(item[0]))
    i=0
    while i<len(events):
        probability=float(events[i][0])
        while i<len(events) and float(events[i][0])==probability:
            _,(q,t)=events[i]
            counts=state[q]
            total-=entity_f05(*counts)
            if t in correct[q]:
                counts[0]+=1
                counts[2]-=1
            else:
                counts[1]+=1
            total+=entity_f05(*counts)
            i+=1
        value=total/len(truth)
        curve.append({'threshold':probability,'macro_f05':value})
        if value>best['macro_f05']+1e-12:
            best={'threshold':probability,'macro_f05':value}
    best['macro_f05']=score(truth,decisions(truth,pairs,probabilities,best['threshold']))['macro_f05']
    return best,curve

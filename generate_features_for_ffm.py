import hashlib, argparse, csv, sys

if len(sys.argv) == 1:
	sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nr_bins', type=int, default=int(1e+6))
parser.add_argument('csv_path', type=str)
parser.add_argument('out_path', type=str)

args = vars(parser.parse_args())

def gen_feats(row):
    feats = []

    features = row.keys()
    count = 0
    for field in features:
        if (field == 'label') or (field == 'conversionTime') or (field == 'instanceID'):
            continue
        value = row[field]
        #field is the feature, count is the index of the feature
        key = field + '-' + str(count) + '-' + str(value)
        feats.append(key)
        count += 1
    return feats

def hashstr(str, nr_bins):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16)%(nr_bins-1)+1

def gen_hashed_fm_feats(feats, nr_bins):
    feats = ['{0}:{1}:1'.format(field, hashstr(feat, nr_bins)) for (field, feat) in feats]
    #feats = ['{0}:{1}:1'.format(field, feat) for (field, feat) in feats]
    return feats

f = open(args['out_path'], 'w')
ln = 0
for row in csv.DictReader(open(args['csv_path'])):
    ln = ln+1
    print 'processing line',ln
    feats = []

    for feat in gen_feats(row):
        #field is the index, feat is the combination of feature and value
        field = feat.split('-')[1]
        feats.append((field, feat))

    feats = gen_hashed_fm_feats(feats, args['nr_bins'])
    f.write((row['label'] if 'label' in row else '') + ' ' + ' '.join(feats) + '\n')


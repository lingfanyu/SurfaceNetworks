#!/usr/bin/env python
import os
import sys
import argparse
import subprocess
import shutil

parser = argparse.ArgumentParser(description='Training Controller, For normal')
parser.add_argument('--model', default='lap')
parser.add_argument('--layer', default='15')
parser.add_argument('--patch', default='1024')
parser.add_argument('--size', default='10k')
parser.add_argument('--num-updates', default='5')
parser.add_argument('--num-epoch', default=300, type=int)
parser.add_argument('--lr', default='1e-3')
parser.add_argument('--gpu-slot', default=0, type=int)
parser.add_argument('--batch-size', default='32')
parser.add_argument('--additional-opt', default='hack1')
parser.add_argument('--deser', default='')
parser.add_argument('--test-dump', default='')
parser.add_argument('--prof', default='')
parser.add_argument('--upsample', default=0)
parser.add_argument('--hidden', default=128)
parser.add_argument('--var-size', action='store_true')
#parser.add_argument('--use-threshold', type=int, default=None)
parser.add_argument('--use-schedule', type=str, default=None)

args = parser.parse_args()

if args.prof != '':
    args.prof = f'nvprof --export-profile {args.prof}'
    #args.prof = f'nvprof --csv --log-file {args.prof}'
if args.test_dump != '':
    des_opt = (os.path.basename(args.deser).split('_'))
    args.size = des_opt[0] +'/'+des_opt[1]
    args.patch = des_opt[2]
    args.model = des_opt[4]
if args.var_size:
    data_path = f'{os.environ["HOME"]}/{args.size}-var/'
    test_path = '@'
else:
    data_path = f'{os.environ["HOME"]}/{args.size}/train/{args.patch}/'
    test_path = f'{os.environ["HOME"]}/{args.size}/test/{args.patch}/'
res = f'{args.size.replace("/","_")}_{args.patch}_{args.lr}_{args.model}'
cmd = f'CUDA_VISIBLE_DEVICES={args.gpu_slot} {args.prof} python3 normal_predict/train_4_normal.py --input V --output normal --model {args.model} --lay {args.layer} --dat {data_path} --test {test_path} --plot 1 --batch {args.batch_size} --num-u {args.num_updates} --lr {args.lr} --res {res} --uni --num-e {args.num_epoch} --half-lr 20  --additional-opt {args.additional_opt} --upsample {args.upsample} --hidden {args.hidden} --no-test --pre --num-u {args.num_updates}'
if args.var_size:
    cmd += ' --var'
    #if args.use_threshold is not None:
    #    cmd += f' --use-threshold {args.use_threshold}'
    sched_path = os.path.abspath(args.use_schedule)
    cmd += f' --use-schedule {sched_path}'

if args.test_dump != '':
    cmd += ' --debug --only-forward-test --dump-dir '+args.test_dump
#else:
#    cmd += ' --no-test'
if args.deser != '':
    cmd += f' --deser {args.deser}'
print(cmd)
subprocess.run(cmd, shell=True)

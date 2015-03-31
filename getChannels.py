#!/usr/bin/env python
import mne
import os
import sys
from __future__ import print_function

import glob

"""
checkfif 
  - returns true only if the fif file exists
  - prints to stderr if file doesn't exist
"""
def checkfif(fif):
  if not os.path.isfile(fif):
    print("cannot find " + fif, file=sys.stderr)
    return False
  else:
    return True
  

# create {channel: ['good', [] ]} DataStruct
def readChannels(fif):
  r = mne.io.Raw(fif)
  channels = [ x.replace('MEG','') \
                for x in r.ch_names   \
                if "MEG" in x  ] 
  return {c: ['good', []] for c in channels }

fifs={};


allchannelsgood={};
# look through all bad channels annotation files
# final output like fif: channel: ['bad|good', ['reson1','reason2',..] ]
for f in glob.glob('/data/Luna1/MultiModal/Clock/*/MEG/*bad*.txt'):
  # rename txt file to fif file
  fif = f.replace('_bad','').replace('.txt','_raw.fif');
  if not checkfif(fif): continue
  # test that fififle exists
  fh = open(f,'r')
  if len(allchannelsgood)<=0:
    allchannelsgood = readChannels(fif)
  fifs[fif] = allchannelsgood;
  for line in fh:
    # break into 'CHID', 'ANNOT1', 'ANNOT2'
    dr = line.rstrip() \
             .upper()  \
             .replace('MEG','') \
             .replace(' ','#').replace('\t','#')   \
             .split('#')
    # remove empty 
    dr = [ x for dr if x ]

    # if we dont have an annot, call it 'bad'
    # set annotations for channel dr[0] of this fif 
    fifs[ fif ][ dr[0] ] = ['bad', dr[1:] ]



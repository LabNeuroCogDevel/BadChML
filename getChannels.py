#!/usr/bin/env python
from __future__ import print_function
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.fftpack import fft,rfft, fftshift, fftfreq
import numpy as np
import mne
import os
import sys
import time

import glob
import random
import re

#learning
from sklearn import svm
from sklearn.metrics import confusion_matrix

#paralell
#from joblib import Parallel, delayed
#from joblib.pool import has_shareable_memory

#vis
import matplotlib.pyplot as plt

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
  

"""
getFeats(rawfif,channelnumber)
 input:
 - rawfif as read by mne.io.Raw (in readFif)
 - channel as it is parsed by readBad (4 digit string)

 get features given a channel
  - extract 4 (nregions) regions that are a 50th (regionsize) of the total data
  - get mean, std, fitted spline residual, coef, and knots

 TODO: rework to use channels some distance away
 TODO: do in parallell??
"""
def getFeats(r,chnldigits):
  chidx=r.ch_names.index('MEG'+chnldigits)

  # define what data to extract
  nregions  = 4
  regionsize= round(r.n_times/50)
  starts    = [round(x * r.n_times/nregions) for x in range(nregions)]


  # init final return
  allsampleFeats = []
  # dont want to hardcode num features or mess with idxs
  #np.empty(5*nregions) 

  for s in [int(x) for x in starts]:
    # extract data
    d,t =  r[chidx,slice(x,x+regionsize)]
    d = d[0,:]*10**10 # make values usable for fitting

    ## SPLINE
    #print("spline")
    # sfitstime = time.time()
    # sfit = UnivariateSpline(t,d,k=5)
    # sfitetime = time.time()
    # sfittime= sfitetime - sfitstime
    ## add to featS:
    #         sfittime, \
    #         sfit.get_residual(),\
    #         sfit.get_coeffs(), \
    #         sfit.get_knots()\

    ##FREQ
    # grab first 10 of fourier series
    # does this makes sense? 
    #
    # print("fft")
    # #fseries = fftshift(rfft(d))
    # fseries = rfft(d)
    # truncfseries = fseries[0:10]

    # plt.plot(d,'.'); plt.plot(sfit(t)); plt.plot(fseries); plt.show()
    

    # the features to use for machine learning
    feats = np.hstack([\
             np.mean(d),\
             np.std(d),\
             np.percentile(np.diff(d),80),\
            ])

    # push features to  
    allsampleFeats.append(feats)

  return np.hstack(np.array(allsampleFeats))


"""
readFif('file_raw.fif',[feature_function])
 - reads in a fif file
 - each channel is a sample
   - labels all as 'good'
   - add features  with feature_function (getFeats by default)
 - returns  {chanelname => {label => ['good',[]], features=>[] } }
"""
# create {channel: ['good', [] ]} DataStruct
def readFif(fif,featfunc=getFeats):
  r = mne.io.Raw(fif)
  channels = [ x.replace('MEG','') \
                for x in r.ch_names   \
                if "MEG" in x  ] 

  return {c: \
            { 'label': ['good', []],\
              'features': featfunc(r,c) \
            } for c in channels }


"""
do a bunch of bandpasses for roc curves
"""
def chFreqs(r,chnldigits):
  chidx=r.ch_names.index('MEG'+chnldigits)
  # extract and make values usable for fitting
  d,t =  r[chidx,:]
  d = d[0,:]*10**10
 
  frr = rfft(d)
  frq = fftfreq(len(t),t[1]-t[0])
  return {'frq': frr,'t': frq}

"""
readBad(badchannelTextfile)
read bad channels from text file
 - formated like MEGchannel #reason1 #reason2
    MEG0000 #D
"""
def readBad(txt):
  fh = open(txt,'r')
  labels=[];
  for line in fh:
    # break into 'CHID', 'ANNOT1', 'ANNOT2'
    dr = line.rstrip() \
             .upper()  \
             .replace('MEG','') \
             .replace(' ','#').replace('\t','#')   \
             .split('#')
    # remove empty 
    dr = [ x for x in dr if x ]
    labels.append( dr )
  return labels


def fifFromTxt(f):
  fif = f.replace('_bad','').replace('.txt','_raw.fif');
  # test that fififle exists. otherwise, whats the point
  if not checkfif(fif): return False
  return fif
"""
readAnnots(bad_channel_textfile, [function_to_annote+getFeatures]) 
  - find fif file assocated with textfile
    expect _bad => '' +  '.txt' => 'raw.fif'
  - get features for all the channels
  - build features and annotations for each channel for each fif
  - mark channels in text file as bad, all others good
"""
def readAnnots(f,LabAndFeatFunc=readFif):
  # rename txt file to fif file
  fif = fifFromTxt(f)
  if not fif: return (fif, False)

  # read features (and default annotation)
  if LabAndFeatFunc: 
    # read in channel names
    # set annotations and features
    annots = LabAndFeatFunc(fif)
  else:
    annots = {}

  #get bad channel labels
  badlabels = readBad(f);
  # if we dont have an explicit annot, call it 'bad'
  # set annotations for channel dr[0] of this fif 
  for dr in badlabels:
    # if we dont have data for this channel (only want bad channel labels)
    if not annots.has_key(dr[0]): annots[dr[0]]={}  
    # label channel bad with annotions (if any)
    annots[ dr[0] ]['label'] = ['bad', dr[1:] ]
  return (fif,annots)

""" 
read through all the bad files to get annotations
- give glob pattern ofr bad txt
- expects raw fif to be named like bad txt (see readAnnots)
- return fifs: dictionary(fif) of dictionaries (label, feature)

"""
def readAll(mx="+inf",\
            pattern='/data/Luna1/MultiModal/Clock/*/MEG/*bad*.txt'):
  fifs={};
  count=0
  # look through all bad channels annotation files
  # final output like fif: channel: ['bad|good', ['reson1','reason2',..] ]
  for i,f in enumerate(glob.glob(pattern)):
    # if we hit max num of fifs we want to read in, stop
    if i>mx: break
    # read in annotations from file f
    (fif,annotes) = readAnnots(f)
    # make sure we found both an annotation file and a raw fif
    # NB: still counts toward reachign mx
    if not annotes: continue
    # add to larger data struct
    fifs[fif] = annotes

  return fifs;

def learn(fifs):
  # ugly way to collapse the fifs data structure
  y = [ v['label'][0]=='good' for f in fifs.values()  for v in f.values() ]
  X = [ v['features'] for f in fifs.values() for v in f.values() ] 

  clsfr = svm.SVC()
  clsfr = clsfr.fit(X,y)
  yhat= clsfr.predict(X)
  cm = confusion_matrix(y,yhat)
  return (clsfr,cm)


"""
plot confusion
mod from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

(clsf,cm) = learn(fifs)
plot_cm(cm,['Bad','Good']).show()
"""
def plot_cm(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    # normalize cm
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # plot
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt

"""
freqsFromList(fif_and_channels)
 - input is array of dicts; like:
    [ {fif=>'filename.fif',chls=>[ ['0000', 'D'] ,['0000','D','HS'],... ]}, {...}, ... ]
 - output is freq and time for each
"""
def freqsFromList(chlist):
  freqs=[]
  for fifch in chlist:
    if not fifch['fif']: continue
    r = mne.io.Raw(fifch['fif'])
    for ch in fifch['chls']:
      if not ch[0]: continue
      freqs.append( {\
          'file':  fifch['fif'],\
          'chnl':  ch,\
          'finfo': chFreqs(r,ch[0]) \
      } )
      #(frr,frq)=chFreqs(r,ch)
  return freqs

"""
put it all together
 not in __main__ so '%run getChannels.py' in ipyton doesn't take forever
"""
def runme():
  ### EXAMPLE LEARNING
  #  -- not good
  # read in 10 annotations/fifs
  samples  = readAll(mx=10)
  # train a classifier with them
  (cls,cm) = learn(samples)
  # plot the confusion matrix
  plot_cm(cm,['Bad','Good']).show()

def getChannels(fif):
    r = mne.io.Raw(fif)
    return [ x.replace('MEG','') \
                  for x in r.ch_names   \
                  if "MEG" in x  ] 

def collapseChnlDS(chds):
  return [ [ x['fif'],y[0], y ]   for x in chds for y in x['chls'] ]

def bandpassFeatures():
  ### find better features
  # sliding bandpass for a whole fif
  # bin frquences, use roc to see whos better where

  # bchfile='/data/Luna1/MultiModal/Clock/11246/MEG/11246_clock_bad_run1.txt'
  # bc=readAnnots(bchfile,lambda x: readFif(x, chFreqs))

  # only look at empty room
  r=re.compile('.*emptyroom.*')
  bdchtxtfiles = filter(r.match, glob.glob('/data/Luna1/MultiModal/Clock/*/MEG/*bad*.txt') )
  bc    = [ {'fif': fifFromTxt(x), 'chls': readBad(x) }  for x in bdchtxtfiles if x ]
  bdfrqs = freqsFromList(bc)

  goodchnls =[];
  for bt in [ x for x in bdchtxtfiles if fifFromTxt(x) ]:
    f= fifFromTxt(bt)
    allch = getChannels(f)
    badch = [ x[0] for x in readBad(bt)]
    goodch = [ [x, ['good']] for x in list(set(allch) - set(badch))  ]
    goodchnls.append({'fif':f, 'chls': goodch})
  
  somegood = [ {'fif':x['fif'], 'chls': random.sample(x['chls'],3)}  for x in goodchnls ]

  collapseChnlDS(bc)
  collapseChnlDS(goodchnls)

  gdfrqs = freqsFromList( somegood )

  



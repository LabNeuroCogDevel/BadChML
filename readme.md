# Bad Channel Detection with Machine Learning
Scores have identified and labeled bad channels in MEG data.

Given these annotations, we want to extract features that enable a ML algorithm to identify bad channels.

## Features

How are these selected? What is important

## Annotations
see `/data/Luna1/MultiModal/Clock/*/MEG/*bad*txt`, 
```bash
perl -lne 'print join("\n",split(/\s+/,uc($1))) if m/.*#(.*)/' /data/Luna1/MultiModal/Clock/*/MEG/*bad*txt|sort |uniq -c
#    254 BC
#    476 D
#    132 HA
#   1375 HF
#      1 HS
#      1 SOP
#   1066 SP
```

|annot|desc|
|-----|----|
| BC  | |
| D   |dead |
| HA  | |
| HF  | |
| SP  | |

## Learning

### compare channels to others in similiar position (Dani)

#!/usr/bin/env perl
use strict; use warnings; use v5.16;

my @files = glob('/data/Luna1/MultiModal/Clock/*/MEG/*bad*.txt');

my @annotes = ();
for my $f (@files) {
  open my $fh, $f or next;
  my $fifname=$f;
  $fifname=~s/_bad//;
  $fifname=~s/.txt$/_raw.fif/;
  if ( ! -r $fifname ) { 
   print STDERR "no $fifname\n";
   next; 
  }
  while(<$fh>) {
     chomp;
     next unless s/(.*)\s*#\s*//;
     my $ch = $1;
     $ch=~s/\s*$//; # why is the extra ' ' still in $1 after regexp above
     $ch=~s/^MEG//; 
     push @annotes, map { "$fifname $ch ". $_ } split /[\s#]+/; 
  }
  close $fh;
}

say join("\n",@annotes);

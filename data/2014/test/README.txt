
SemEval 2014 Task 8: Broad-Coverage Semantic Dependency Parsing: Test Data

Version 1.1; March 31, 2014


Overview
========

This directory contains three files with semantic dependency graphs, one file
per SDP target format (DM, PAS, and PCEDT), together with three variants of
these files as unannotated parser inputs, and two files providing ‘companion’
syntactic analyses for optional use in the open track of the task.

The test data comprises 1348 sentences (or 29808 tokens) of text taken from
Section 21 of the PTB WSJ Corpus.  The three input files use the extension
‘.tt’ because they only provie the first four columns of the SDP tab-separated
file format, viz.

  (1) running identifier
  (2) token surface form
  (3) format-specific lemma
  (4) PTB-style part of speech

It is expected that participating systems use the ‘.tt’ files as input files,
effectively adding to each non-empty line columns (5), (6), and upwards.  Thus,
system outputs must be submitted in the official SDP file format (and using the
common suffix ‘.sdp’), one file per target format, as documented on-line:

  http://sdp.delph-in.net/2014/data.html

The three annotation formats have been aligned sentence- and token-wise, i.e.
they annotate the exact same text; however, there are differences in LEMMA and
POS fields.  Nevertheless, for each format the conventions used in these fields
are exactly the same as were used in the corresponding training data.

Using the gold-standard semantic dependency graphs (with suffix ‘.sdp’) that
are included in this re-release of the test data (after the evaluation period
of the 2014 task has expired), participants can score their own system results
using the official scorer:

  https://bitbucket.org/kuhlmann/sdp

For submissions to the ‘open’ track (where it is legitimate to use additional
resources or tools, beyond the task-specific training data), we make available
the same two formats of syntactic analyses, using the same tools and setup, as
described in detail in the SDP companion training data:

  http://svn.delph-in.net/sdp/public/companion/current.tgz

The only difference in the companion test data provided as part of this archive
(i.e. the two files with the common suffix ‘.cpn’) is the training data used:
our companion syntactic analyses for the SDP test data result from parsing the
pre-tokenized text with parser models trained on the complete set of sentences
from WSJ Sections 00–20 used as training data in the SDP context.


Known Errors
============

None, for the time being.


Release History
===============

[Version 1.1; March 31, 2013]

+ Re-release of test and companion data, now including gold-standard targets.

[Version 1.0; March 21, 2013]

+ Initial release of test and companion data in three formats: DM, PAS, PCEDT.


Contact
=======

For questions or comments, please do not hesitate to email the task organizers
at: ‘sdp-organizers@emmtee.net’.

Dan Flickinger
Jan Hajič
Marco Kuhlmann
Yusuke Miyao
Stephan Oepen
Yi Zhang
Daniel Zeman

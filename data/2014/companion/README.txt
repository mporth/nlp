
SemEval 2014 Task 8: Broad-Coverage Semantic Dependency Parsing: Companion Data

Version 1.1; March 07, 2014


Overview
========

This directory contains the so-called ‘companion’ data: syntactic analyses in
various (more or less) popular formats, for use in the open track of the task.

In all cases, we seek to make available realistic outputs of state-of-the-art
tools.  Specifically, we applied ten-fold jack-knifing over the SDP training
data, splitting off nine of ten sentences for training and the remaining ten
percent for testing, ten times.  For each split, we re-trained the syntactic
analyser on the training portion and applied the resulting model to the test
portion.  Thus, in contrast to using off-the-shelf models for these parsers,
our companion data reflects typical parser performance on unseen, in-domain
test data (each parser is trained on some 30,000 sentences, i.e. a little less
than is common in statistical parser evaluation over the PTB).  As our SemEval
2014 task moves into its evaluation period, we will make available the same
range of companion analyses (using models trained over the full SDP training
data) for the SDP test data.

Please see below for the individual companion formats and details on how they
were prepared.  In all cases, the companion data is intended solely for use in
the open track, where one can think of the additional files as extending our
‘core’ files (‘dm.sdp’, ‘pas.sdp’, and ‘pcedt.sdp’) ‘horizontally’, i.e. with
additional columns.  For convenience of reference we will number these columns
using negative identifiers, starting from -1 and working backwards.  Companion
files are sentence- and token-aligned to our training data.


Simplified PTB-Style Phrase Structure Trees
===========================================

In a mimicry of the CoNLL 2004 and 2005 Shared Tasks, we provide PTB-style
phrase structure trees with the ‘standard’ range of simplifications applied,
viz. removal of function labels and traces.  We trained the Berkeley Parser
(Petrov & Klein, 2007) on the simplified PTB trees corresponding to (90% of)
the SDP training data (in each fold), and then applied the resulting grammars
to pre-tokenized input corresponding to the remaining 10% (in other words, we
did not provide PoS tags in parser inputs).  Parser outputs were converted to
the token-oriented so-called Start-End (SE) Format of CoNLL 2005, using the
original scripts from the 2005 Shared Task.  For more background, see:

  https://code.google.com/p/berkeleyparser/
  http://www.lsi.upc.edu/~srlconll/

We largely used default settings in training and parsing, though turned on the
‘-accurate’ switch to the parser because it reduced parser failures, i.e. the
number of strings for which the parser did not produce an output.  In the end,
two sentences could not be parsed (six without ‘-accurate’), and we padded the
companion file with empty values (‘_’) for the tokens of these sentences.

Parser training and decoding were invoked as follows:

  java -Xmx8G -cp BerkeleyParser-1.7.jar \
    edu.berkeley.nlp.PCFGLA.GrammarTrainer \
    -path fold${i}-train.mrg -out fold${i}-train.grammar -treebank SINGLEFILE
  
   java -Xmx4G -jar BerkeleyParser-1.7.jar \
     -gr fold${i}-train.grammar -accurate \
     < fold${i}-parse.txt > fold${i}-berkeley.mrg;

The companion file ‘sb.berkeley.cpn’ provides two fields:

  (-1) BERKELEYPOS --- PoS tags predicted by the parser
  (-2) BERKELEYSE  --- bracketing predicted by the parser


Stanford Basic Dependencies
===========================

To provide syntactic analyses in the form of bi-lexical dependencies, we used
the so-called Stanford Basic scheme (de Marneffe & Manning, 2008) and the
parser of Bohnet & Nivre (2012).  The conversion started from the exact same
simplified PTB-style files used in connection with the Berkeley Parser above;
we applied the dependency converter built into the Stanford Parser (in version
3.3.1) as follows:

  java -cp stanford-parser.jar \
    edu.stanford.nlp.trees.EnglishGrammaticalStructure \
    -treeFile fold${i}-train.mrg -basic -keepPunct -conllx \
    > $(basename fold${i}-train .mrg).conll

Subsequently, the CoNLL-X format output by the converter was padded with extra
columns, to provide the CoNLL-09 format expected by the Bohnet & Nivre Parser
(henceforth BN): for the training files, columns 1 (ID), 2 (FORM), 4 (POS), 7
(HEAD), and 8 (DEPREL) were preserved; for the decoding files, only columns 1
and 2 were provided to the parser (thus, again, the parser did not have access
to gold-standard parts of speech at decoding time).

Parser training and decoding were invoked as follows (in one run per fold):

  java -Xmx30G -cp anna-3.3.jar is2.transitionR6j.Parser \
    -train fold${i}-train.conll \
    -test fold${i}-parse.conll -out fold${i}-bn.conll \
    -model eng-b80-R6j-${i}.mdl -tmodel eng-b80-R6j-${i}.tmdl \
    -i 20 -hsize 500000001 -beam 80 -1st a -2nd abcd -3rd ab \
    -tsize 3 -tnumber 10 -ti 10 -x train:test \
    -thsize 90000001 -tthreshold 0.2 -tx 2 -half 2 -tt 25 -cores 16

For more background, please see:

  https://code.google.com/p/mate-tools/
  http://nlp.stanford.edu/software/stanford-dependencies.shtml
  http://www-nlp.stanford.edu/software/lex-parser.shtml

The companion file ‘sb.bn.cpn’ provides three fields:

  (-3) BNPOS    --- PoS tags predicted by the parser
  (-4) BNHEAD   --- head identifiers predicted by the parser
  (-5) BNDEPREL --- dependency labels predicted by the parser


Known Errors
============

None, for the time being.


Release History
===============

[Version 1.1; March 7, 2014]

+ Eliminate a handful of ‘none’ (invalid) dependencies in B&N parser outputs;
+ drop superfluous line breaks between tokens (i am surprised nobody complained).

[Version 1.0; January 17, 2014]

+ Initial release of the SDP companion data in PTB and Stanford Basic formats.


Contact
=======

For questions or comments, please do not hesitate to email the task organizers
at: ‘sdp-organizers@emmtee.net’.

Marco Kuhlmann
Stephan Oepen

# srilm_format.py
### convert the debug output from srilm 'ngram' to tabular format

### sample of debug 2 output:
# k h m e r
# 	p( k | <s> ) 	= [2gram] 0.00984863 [ -2.00662 ]
# 	p( h | k ...) 	= [2gram] 0.00119095 [ -2.92411 ]
# 	p( m | h ...) 	= [2gram] 0.00628931 [ -2.2014 ]
# 	p( e | m ...) 	= [2gram] 0.146143 [ -0.835221 ]
# 	p( r | e ...) 	= [2gram] 0.138419 [ -0.858805 ]
# 	p( </s> | r ...) 	= [2gram] 0.156905 [ -0.804362 ]
# 1 sentences, 5 words, 0 OOVs
# 0 zeroprobs, logprob= -9.63052 ppl= 40.2797 ppl1= 84.3535

# a b w a t
# 	p( a | <s> ) 	= [2gram] 0.0639249 [ -1.19433 ]
# 	p( b | a ...) 	= [2gram] 0.0302599 [ -1.51913 ] # this is exactly the listed logprob for 'a b', p(b | a)
# 	p( w | b ...) 	= [1gram] 0.00093305 [ -3.0301 ] # this value is what? unattested 'b w', the BOW for 'b' and 'w' are ~ -0.9; the sum of their own logprobs is -3.8
# 	p( a | w ...) 	= [2gram] 0.189853 [ -0.721583 ]
# 	p( t | a ...) 	= [2gram] 0.130234 [ -0.885275 ]
# 	p( </s> | t ...) 	= [2gram] 0.274375 [ -0.561656 ]
# 1 sentences, 5 words, 0 OOVs
# 0 zeroprobs, logprob= -7.91207 ppl= 20.8295 ppl1= 38.2309

# file debug_ortho_spaced.txt: 3 sentences, 15 words, 0 OOVs
# 0 zeroprobs, logprob= -23.6476 ppl= 20.5946 ppl1= 37.7142

### in debug 2, words may be variable length and therefore have variable number of lines
### words are newline delimited
### output is finished with a file summary

### debug 1 output:
# a b a c u s
# 1 sentences, 6 words, 0 OOVs
# 0 zeroprobs, logprob= -7.43736 ppl= 11.5473 ppl1= 17.3604

# file debug_ortho_spaced.txt: 3 sentences, 15 words, 0 OOVs
# 0 zeroprobs, logprob= -23.6476 ppl= 20.5946 ppl1= 37.7142

### in debug 1, each item has exactly 3 lines: sentence, counts, stats
### this version is easier to get logprobs from, I guess.

### Even easier to first collapse newlines?
	## 1. remove \n before '1 sentences'
	## 2. remove \n before '0 zeroprobs'
	## 3. remove footer (now 1 line, begins with 'file ')
	## 4. replace \n\n with \n
	## 5. replace ' ppl' with ', ppl'
	## 5. remove linejunk: 'sentences, words, OOVs, zeroprobs, logprob=, ppl=, ppl1='
	## 6. add header 'item, sentences, words, OOVs, zeroprobs, logprob, ppl, ppl1'
	## 7. replace ', ' with ','
		## or, just remove all spaces
### How can this be done with multiline regex in Python?

import re

## read in the file; specify filename here
	## Eventually, make this a shell argument.
baseFilename = 'SRILM_PPL_output.txt'

## Output suffix can be anything, but this makes it clear.
outFilename = baseFilename[:-4]+'_ppl.csv'

baseFile = open(baseFilename, 'r')
outFile = open(outFilename, 'w')

baseWhole = baseFile.read()
baseFile.close()

# string.replace() is not regex
# regex is: re.sub(pattern, repl, string, max=0)

# remove the chunk-internal newlines
baseWhole = baseWhole.replace('\n1 sentences','')
baseWhole = baseWhole.replace('\n0 zeroprobs','')

# remove the footer; need regex for this
baseWhole = re.sub(r'\nfile.*\n', '', baseWhole)

# remove all the label junk from the Whole file
baseWhole = baseWhole.replace('words',"")
baseWhole = baseWhole.replace('0 OOVs, ',"")
baseWhole = baseWhole.replace('logprob=',"")

# remove ppl labels but add commas
baseWhole = baseWhole.replace('ppl=',",")
baseWhole = baseWhole.replace('ppl1=',",")

# remove all spaces
baseWhole = baseWhole.replace(' ','')

# break into lines (not strictly necessary)
baseChunks = baseWhole.split('\n\n')

# write out file 
	# ppl and ppl1 are not what we're usually interested in, but might as well preserve them.
fileHeader = 'item,length,logprob,ppl,ppl1'
outFile.write(fileHeader+'\n')
for line in baseChunks:
	outFile.write(line+'\n')
outFile.close()

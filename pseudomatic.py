### pseudomatic.py

'Last Modified: 2017-3-17'
'Jeremy M. Needle; Northwestern University, Department of Linguistics'
'Wordovators Project; PI: Janet B. Pierrehumbert, University of Oxford'

'Description: This tool generates pseudoword strings based on corpus-learned phonotactics, using a Finite State Machine from a text sample. The FSM is built on n-graph statistics for n >= 2. The elements are symbols (single ASCII letters). Input corpus may be phonemic or orthographic, but each element should be a single character (so the phonemic alphabet must be monosymbolic, like DISC).'
'	Output may include real words, including those in the training corpus.'
'	Functions are available to compute the harmonic mean transitional probability for words (as sum of logs), as a "goodness" score for the pseudowords.'

'Input: Corpus files should be plaintext (.txt) files with one word per line (\n line endings), no whitespace.'

'Output: Multiple outputs are available (.txt and .csv); see specific functions for details.'

'Methods:'
'fsm.	# all functions are members of the fsm class'
'	read	# populates the FSMs with blocks from an input file'
'	shift	# returns the list of successors to a string'
'	transProb	# builds the transition dictionary from block to blocks based on the FSM, along with transitional probabilities'
'	genPath	 # subroutine: generates a random word of given length'
'	pathToString	# subroutine: converts a FSM path (sequence of blocks) to a string'
'	sampleWords	 # Uses random method to return the given number of words of given length from the FSM'
'	wordScore	# subroutine: computes the harmonic mean transitional probability for a word given the FSM'
'	scoreWords	# computes wordscore for each word in a list and saves the output to CSV; illegal words are assigned score 0'
'	scoreTransform # subroutine: transforms probabilities using different functions'
'Usage:'
'	example = pseudomatic.fsm("input_corpus.txt")'
'      # initialize fsm by reading the input text file'
'	example.transProb()'
'		# store 1,2,3-gram statistics; print phonotactic type information to console'
'	example.sampleWords(6,3,80)'
'      # generate using 3-grams, 80 words of length 6, saved to "wordset_6_sample_80.txt" and "wordsetS_6_sample_80.txt"'
'	example.sampleFillers(6,100, weighted=True)'
'      # generate using weighted 1-grams, 100 fillers of length 6, saved to "fillers_6_sample_100_weighted.txt"'
'	example.scoreWords("wordsetS_6_sample_80.txt")'
'      # process input file by adding probability scores, saved to "wordsetS_6_sample_80_scored.csv" file'

import random, math, operator, bisect, numpy

class fsm:
	'This class implements a custom paired Finite State Machine for phonotactics, both order 2 and 3 together.'
	
	def __init__(self, filename = None):

		'Initialize the FSM, and train on filename if supplied'
		# Save blockwidths as constants
		self.DIPHONE_SIZE = 2
		self.TRIPHONE_SIZE = 3
		
		# Store two blocksets separately
		self.diBlockset = {}
		self.triBlockset = {}
		self.unigrams = {}
		
		#* these can stay the same
		self.start,self.end = '\0','\0'
		
		# if a filename was supplied, train on it. Otherwise you will need to type in the blocks
		if filename:
			self.read(filename)

	def read(self, filename, reset = True):
		'Finds all blocks that occur in filename and make them legal'
		'Can be re-used; defaults to erasing existing blocksets'
		'Blocks are organized into a Python dictionary'
		'Frequency of blocks in the training corpus is saved in the dict'
		if reset:
			
			self.diBlockset = {}
			self.triBlockset = {}
			self.unigrams = {}
			self.wordCount = 0
				
		f = open(filename, 'r')

		# Iterate over each line in the file (one word per line, Unix line-endings)
		for line in f:
			
			# Skip empty lines ( \n )
			if line == '\n': continue

			# If line wasn't empty, increment the word counter.
			self.wordCount += 1

			# Format the word: strip newline char, add null ( \x00 ) at both ends
			augStr = self.start + line[:-1] + self.end
			
			# Read current line into the fsm_3
			for blockOffset in xrange(len(augStr)-self.TRIPHONE_SIZE+1):
				curBlock = augStr[blockOffset:(blockOffset+self.TRIPHONE_SIZE)]
				if curBlock in self.triBlockset:
					self.triBlockset[curBlock] += 1
				else:
					self.triBlockset[curBlock] = 1

			# Read current line into the fsm_2
			for blockOffset in xrange(len(augStr)-self.DIPHONE_SIZE+1):
				curBlock = augStr[blockOffset:(blockOffset+self.DIPHONE_SIZE)]
				if curBlock in self.diBlockset:
					self.diBlockset[curBlock] += 1
				else:
					self.diBlockset[curBlock] = 1

			# Read current line into the unigram set
			for phone in augStr:
				if phone in self.unigrams:
					self.unigrams[phone] += 1
				else:
					self.unigrams[phone] = 1
		f.close()


	def shift(self, string, size, order): 
		'Returns list of all blocks that match string after it has been shifted'
		'	e.g. shift("ab") might return ["bc", "bd"]'
		'	this would mean that (only) "c" and "d" are legal successors of "b"'

		## Note: lookups using self.transDict2 or self.transDict3 are faster, if it's already been constructed. Construction uses shift().
		## E.g., self.transDict2['\x00'].keys() returns the block1set.
		## E.g., self.transDict2[block].keys() returns the successor list.

		# initialize an empty list to hold successors
		successors = []
		matchStr = string[size:]
		# the substring on which string and successor must match
		# "ab" ==> "b"
		
		# Link the order-appropriate self.transDict to the local var transDict
		if order == 2:
			blockset = self.diBlockset
		elif order == 3:
			blockset = self.triBlockset
		else:
			raise Exception('shift() needs a valid order argument (2 or 3).')

		# test every legal block
		for block in blockset:
			# if end of string matches beginning of block
			if block[:len(matchStr)] == matchStr:	successors.append(block)
			# "bc" ==> "b"
			# then add it to the list of possible successors
		return(successors) 

	def pathToString(self,path):
		'This function converts block paths to string words'
		'Used in: sampleWords()'
		# e.g., ['\x00st','sta','tar','ar\x00'] ==> 'star'

		#initialize string to first block
		outString = path[0]
		
		#strip thatblock
		path = path[1:]
		
		#build string by adding just the last chars of each block
		for block in path:
			outString += block[-1]

		#strip off the start and end nulls
		return outString[1:-1]

	def transProb(self):
		'Uses self.diBlockset and self.triBlockset to calculate transitional probabilities for each successor'
		'Saves to new dicts, "self.transDict2", "self.transDict3".'
		'Used in: wordScore(), genPath() (==> sampleWords()).'

		## transProb() generates and saves both self.transDict2 and self.transDict3 in parallel.

		#* Begin Unigram Section
		# Note that unigrams cannot be transitional, so this just gives relative frequencies
		# Reported in descending order of frequency along with cumulative distribution, for use with weight random choice.
		
		# Create a sorted list of keys from self.unigrams
		phones = sorted(self.unigrams, key=self.unigrams.get, reverse=True)

		# Get the sum of all unigram frequencies (should be pretty large)
		totFreq = sum(self.unigrams.values())

		# For each phone, save the relative frequency (freq/totFreq)
		tempRelFreqs = []
		tempCumWeights = []
		runningTotal = 0
		for phone in phones:
			phoneRelFreq = float(self.unigrams[phone])/totFreq
			tempRelFreqs.append(phoneRelFreq)
			runningTotal += phoneRelFreq
			tempCumWeights.append(runningTotal)

		# Zip the new relative freqs with the matching phones
		# List version with cumulative distribution, for weighted random choice
		self.unigramRelFreqs = zip(phones, tempRelFreqs, tempCumWeights)

		# Dict version, for lookup when scoring
		self.unigramRelDict = dict(zip(phones,tempRelFreqs))

		# Optional console output:
		# Compute the total number of values in the dict:
		count1 = len(self.unigrams)
		print "self.unigramRelDict:"
		print "    ",self.wordCount,"input words,",totFreq,"phone tokens,",count1,"phone types."

		#* End Unigram Section

		#* Diphone section:
		# Save the sorted, stable list version of the block2set (i.e., the start blocks) to a global self.block2set variable; this is used by generation functions.
			
		# First, creates the non-transitional frequency dictionary.
			# Note that start and end blocks have to be calculated differently:
			# Total start blocks == total end blocks == count of words == self.wordCount; this is only true for TOKENS
			# Total medial blocks == total blocks - (total start blocks + total end blocks) == total blocks - count of words * 2 ==  totFreq - 2 * self.wordCount

		# Convert self.diBlockset to digramRelFreqs and digramRelDict
		# Create a sorted list of keys from self.diBlockset
		diBlocks = sorted(self.diBlockset, key=self.diBlockset.get, reverse=True)
		
		# Get the sum of all digram frequencies
		totFreq = sum(self.diBlockset.values())

		# Derive the start/middle/end block total frequencies
		startFreq = self.wordCount
		endFreq = self.wordCount
		medialFreq = totFreq - (2 * self.wordCount)

		# Separate out the start/medial/end blocksets
		blocksetStart = {}
		# Check each diBlock (type); diBlocks is a list of the types, so there should be no repeats to worry about
		for diBlock in diBlocks:
			# If the diBlock ends with self.start, add it to the blocksetStart; note this is a token dictionary, not just type, so the value is looked up from the diBlockset dict.
			if diBlock[0] == self.start:
				blocksetStart[diBlock] = self.diBlockset[diBlock]

		# Note that this method would match a string of '\x00\x00' as both a start and an end; this is not possible for digrams, but can happen for trigrams.
		
		blocksetEnd = {}
		
		# Check each diBlock
		for diBlock in diBlocks:
			# If the diBlock ends with self.end, add it to the blocksetEnd; note this is a token dictionary, not just type, so the value is looked up from the diBlockset dict.
			if diBlock[-1] == self.end:
				blocksetEnd[diBlock] = self.diBlockset[diBlock]

		# blocksetMedial is anything not in blocksetStart or blocksetEnd; use set comparisons?
		blocksetMedial = {}
		for diBlock in diBlocks:
			if not (diBlock in blocksetStart or diBlock in blocksetEnd):
				blocksetMedial[diBlock] = self.diBlockset[diBlock]

		# Optional console output:
		# Compute the total number of values in the dict:
		totDiBlocksTypes = len(self.diBlockset.keys()) # TYPES
		totDiBlocksTokens = sum(self.diBlockset.values()) # TOKENS

		diBlocksStartTypes = len(blocksetStart.keys()) # this gives the number of TYPES
		diBlocksStartTokens = sum(blocksetStart.values()) # this gives the number of TOKENS

		diBlocksEndTypes = len(blocksetEnd.keys()) # types
		diBlocksEndTokens = sum(blocksetEnd.values()) # tokens

		diBlocksMedialTypes = len(blocksetMedial.keys()) # types
		diBlocksMedialTokens = sum(blocksetMedial.values()) #tokens

		# debug stuff
		self.diBlocksetEnd = blocksetEnd


		print "self.digramRelDict:"
		print "    ",self.wordCount,"input words,",totDiBlocksTokens,"total diBlock tokens,",totDiBlocksTypes,"total diBlock types."
		print "    ",diBlocksStartTokens,"diBlock start tokens,",diBlocksStartTypes,"diBlock start types."
		# print blocksetStart
		print "    ",diBlocksEndTokens,"diBlock end tokens,",diBlocksEndTypes,"diBlock end types."
		# print blocksetEnd
		print "    ",diBlocksMedialTokens,"diBlock medial tokens,",diBlocksMedialTypes,"diBlock medial types."
		# print blocksetMedial
		
		# track freq,totFreq pairs in this temp dict until later
		tempDigramRelDict = {}
		for diBlock in diBlocks:
			# assign the appropriate denominator for start, end, and medial blocks
			if diBlock in blocksetStart:
				categoryFreq = diBlocksStartTokens
			elif diBlock in blocksetEnd:
				categoryFreq = diBlocksEndTokens
			else:
				categoryFreq = diBlocksMedialTokens
			
			# dict value is tuple (block freq, relevant totFreq, relFreq, log relFreq, sqrt relFreq)
			relFreq = float(self.diBlockset[diBlock])/categoryFreq
			tempDigramRelDict[diBlock] = (self.diBlockset[diBlock],categoryFreq,relFreq,math.log(relFreq),math.sqrt(relFreq))
		# save the temp dict to the object attribute for later reference
		self.digramRelDict = tempDigramRelDict

		# Set up local temp list of block1 items.
		block1set = sorted(self.shift(self.start, 0, self.DIPHONE_SIZE), key=self.diBlockset.get, reverse=True)
		# this is the same as blocksetStart.keys()
		
		# Initialize temp storage dictionary.
		tempDigramTransDict = {}	

		# First, get the transitions for the null-start (successors = the start blocks)
		# Their transitional frequency is the same as their basic frequency in-category; self.digramRelDict[diBlock]
		# Create a list of frequencies for all the null-to-startblocks transitions
		freqList = []
		for diBlock in block1set:

			freqList.append(self.digramRelDict[diBlock][0])
			tempSum = float(sum(freqList))
			
			# Calculate a vector of the relative frequencies, in order
			relFreqs = [x/tempSum for x in freqList]
			logRelFreqs = [math.log(x) for x in relFreqs]
			sqrtRelFreqs = [math.sqrt(x) for x in relFreqs]
			# construct a list of the same denominator repeated in order to zip together
			denominators = [sum(freqList)] * len(freqList)

			valueList = zip(freqList,denominators,relFreqs,logRelFreqs,sqrtRelFreqs)

			keyList = block1set
			# valueList = zip(freqList,relFreqs,logRelFreqs,sqrtRelFreqs)
			valueList = zip(freqList,denominators,relFreqs,logRelFreqs,sqrtRelFreqs)
			# tempSuccessorDict = {}
			tempDigramTransDict['\x00'] = dict(zip(keyList,valueList))

		# Now that the transitions from null-start to start blocks is done, get the transitions from all non-end blocks to their successors
		# This means that the transDict does not contain top entries for end blocks; that's fine, the score functions stop at the penultimate block because that's the last transition (e.g. 'abc' >> 'bc\x00')

		# construct a list of non-start blocks
		blocksetStartMedial =  blocksetStart.keys() + blocksetMedial.keys()
		# print blocksetStartMedial

		for diBlock in blocksetStartMedial:
			# For each, get the successor list (ordered) with shift
			successors = self.shift(diBlock,1,self.DIPHONE_SIZE)
			
			# Create a list of frequencies for all successors
			freqList = []
			
			# For each successor, save the block frequency to the list; only the block frequency matters for the transition
			for successor in successors:
				freqList.append(self.digramRelDict[successor][0])
			# Sum the frequencies to calculate relative frequencies
			# tempSum = sum(freqList)
				# float for the relative calculation later
			tempSum = float(sum(freqList))
			
			# Calculate a vector of the relative frequencies, in order
			relFreqs = [x/tempSum for x in freqList]
			logRelFreqs = [math.log(x) for x in relFreqs]
			sqrtRelFreqs = [math.sqrt(x) for x in relFreqs]
			# construct a list of the same denominator repeated in order to zip together; sum(freqList) so not float
			denominators = [sum(freqList)] * len(freqList)

			# Match successors and relative frequency pairs and save to a dict
			# tempDigramTransDict[diBlock] = dict(zip(successors,freqList,relFreqs,logRelFreqs,sqrtRelFreqs))
			keyList = successors
			valueList = zip(freqList,denominators,relFreqs,logRelFreqs,sqrtRelFreqs)
			# tempSuccessorDict = {}
			tempDigramTransDict[diBlock] = dict(zip(keyList,valueList))

		# Save out the transDict to self.transDict2 for global use
		# self.transDict2 = tempDigramTransDict
		self.digramTransDict = tempDigramTransDict

		# Optional console output:
		# Compute the total number of values (transitions) in the dict:
		count2 = sum(len(item) for item in self.digramTransDict.itervalues())
		print "self.digramTransDict:"
		print "    ",len(self.diBlockset),"blocks with",count2,"transitions."
		
		self.digramTransList = []
		# unpack into sding and dict
		for key1, transitions in self.digramTransDict.items():
			for key2, value in transitions.items():
				self.digramTransList.append([key1+':'+key2]+list(value))
		self.digramTransListSort = sorted(self.digramTransList, key=lambda transition: transition[3])
		# print 'Minimum digram transition: ',self.digramTransListSort[0]
		#* End diphone half.
		
		#* Triphone half:
		
		# Save the sorted, stable list version of the block2set (i.e., the start blocks) to a global self.block2set variable; this is used by generation functions.

		block1set = sorted(self.shift(self.start, 0, self.TRIPHONE_SIZE), key=self.triBlockset.get, reverse=True)
		
		# Convert self.triBlockset to trigramRelFreqs and trigramRelDict
		# Create a sorted list of keys from self.triBlockset
		triBlocks = sorted(self.triBlockset, key=self.triBlockset.get, reverse=True)
		
		# Derive the start/middle/end block total frequencies
		startFreq = self.wordCount
		endFreq = self.wordCount
		medialFreq = totFreq - (2 * self.wordCount)

		# Separate out the start/medial/end blocksets
		blocksetStart = {}
		blocksetEnd = {}
		blocksetMedial = {}

		# Combined if version:
		for triBlock in triBlocks:
			# If the triBlock ends with self.start, add it to the blocksetStart; note this is a token dictionary, not just type, so the value is looked up from the triBlockset dict.
			# This should give the same result as the block1set method: block1set = sorted(self.shift(self.start, 0, self.TRIPHONE_SIZE), key=self.triBlockset.get, reverse=True)
			if triBlock[0] == self.start:
				blocksetStart[triBlock] = self.triBlockset[triBlock]

			# If the triBlock ends with self.end, add it to the blocksetEnd; note this is a token dictionary, not just type, so the value is looked up from the triBlockset dict.
			# Note that a block that is both start and end (1-letter words like 'a') will be counted as the first match (i.e., a start block)
			elif triBlock[-1] == self.end:
				blocksetEnd[triBlock] = self.triBlockset[triBlock]

			else:
				blocksetMedial[triBlock] = self.triBlockset[triBlock]

		# Optional console output:
		# Compute the total number of values in the dict:
		totTriBlocksTypes = len(self.triBlockset.keys()) # TYPES
		totTriBlocksTokens = sum(self.triBlockset.values()) # TOKENS

		triBlocksStartTypes = len(blocksetStart.keys()) # this gives the number of TYPES
		triBlocksStartTokens = sum(blocksetStart.values()) # this gives the number of TOKENS

		triBlocksEndTypes = len(blocksetEnd.keys()) # types
		triBlocksEndTokens = sum(blocksetEnd.values()) # tokens

		triBlocksMedialTypes = len(blocksetMedial.keys()) # types
		triBlocksMedialTokens = sum(blocksetMedial.values()) #tokens

		print "self.trigramRelDict:"
		print "    ",self.wordCount,"input words,",totTriBlocksTokens,"total triBlock tokens,",totTriBlocksTypes,"total triBlock types."
		print "    ",triBlocksStartTokens,"triBlock start tokens,",triBlocksStartTypes,"triBlock start types."
		# print blocksetStart
		print "    ",triBlocksEndTokens,"triBlock end tokens,",triBlocksEndTypes,"triBlock end types."
		# print blocksetEnd
		print "    ",triBlocksMedialTokens,"triBlock medial tokens,",triBlocksMedialTypes,"triBlock medial types."
		# print blocksetMedial

		# track freq,totFreq pairs in this temp dict until later
		tempTrigramRelDict = {}
		for triBlock in triBlocks:
			# assign the appropriate denominator for start, end, and medial blocks
			if triBlock in blocksetStart:
				categoryFreq = triBlocksStartTokens
			elif triBlock in blocksetEnd:
				categoryFreq = triBlocksEndTokens
			else:
				categoryFreq = triBlocksMedialTokens
			
			# dict value is tuple (block freq, relevant totFreq, relFreq, log relFreq, sqrt relFreq)
			relFreq = float(self.triBlockset[triBlock])/categoryFreq
			tempTrigramRelDict[triBlock] = (self.triBlockset[triBlock],categoryFreq,relFreq,math.log(relFreq),math.sqrt(relFreq))
		# save the temp dict to the object attribute for later reference
		self.trigramRelDict = tempTrigramRelDict
			# tested and looks good. Now, build the transitional dict based on the raw counts in this dict.

		# Initialize temp storage dictionary.
		tempTrigramTransDict = {}
		
		# First add start blocks separately, because they are successors of the invalid null block
		# Their transitional frequency is the same as their basic frequency in-category; self.digramRelDict[triBlock]
		# Create a list of frequencies for all the start blocks
		freqList = []
		for triBlock in block1set:
			freqList.append(self.trigramRelDict[triBlock][0])

			# Sum the frequencies to calculate relative frequencies
			# float for the relative calculation later
			tempSum = float(sum(freqList))
			
			# Calculate a vector of the relative frequencies, in order
			relFreqs = [x/tempSum for x in freqList]
			logRelFreqs = [math.log(x) for x in relFreqs]
			sqrtRelFreqs = [math.sqrt(x) for x in relFreqs]
			# construct a list of the same denominator repeated in order to zip together
			denominators = [sum(freqList)] * len(freqList)

			valueList = zip(freqList,denominators,relFreqs,logRelFreqs,sqrtRelFreqs)

			# Match successors and relative frequency pairs and save to a dict
			keyList = block1set
			valueList = zip(freqList,denominators,relFreqs,logRelFreqs,sqrtRelFreqs)
			tempTrigramTransDict['\x00'] = dict(zip(keyList,valueList))
		
		# construct a list of non-end blocks; tested this OK;
		blocksetStartMedial =  blocksetStart.keys() + blocksetMedial.keys()
		# print blocksetStartMedial

		# Iterate through each block in the blocksetMedialEnd
		for triBlock in blocksetStartMedial:
			# For each, get the successor list (ordered) with shift
			successors = self.shift(triBlock,1,self.TRIPHONE_SIZE)
			
			# Create a list of frequencies for all successors
			freqList = []
			
			# For each successor, save the block frequency to the list; only the block frequency matters for the transition
			for successor in successors:
				freqList.append(self.trigramRelDict[successor][0])
			# Sum the frequencies to calculate relative frequencies
			# tempSum = sum(freqList)
				# float for the relative calculation later
			tempSum = float(sum(freqList))
			
			# Calculate a vector of the relative frequencies, in order
			relFreqs = [x/tempSum for x in freqList]
			logRelFreqs = [math.log(x) for x in relFreqs]
			sqrtRelFreqs = [math.sqrt(x) for x in relFreqs]
			# construct a list of the same denominator repeated in order to zip together; sum(freqList) so not float
			denominators = [sum(freqList)] * len(freqList)

			# Match successors and relative frequency pairs and save to a dict
			# tempDigramTransDict[triBlock] = dict(zip(successors,freqList,relFreqs,logRelFreqs,sqrtRelFreqs))
			keyList = successors
			valueList = zip(freqList,denominators,relFreqs,logRelFreqs,sqrtRelFreqs)
			# tempSuccessorDict = {}
			tempTrigramTransDict[triBlock] = dict(zip(keyList,valueList))	
		
		# Save out the transDict to self.transDict for global use
		# self.transDict3 = tempTrigramTransDict
		self.trigramTransDict = tempTrigramTransDict
		
		# Optional console output:
		# Compute the total number of values (transitions) in the dict:
		count3 = sum(len(item) for item in self.trigramTransDict.itervalues())
		print "self.trigramTransDict:"
		print "    ",len(self.triBlockset),"blocks with",count3,"transitions."

		self.trigramTransList = []
		# unpack into string and dict
		for key1, transitions in self.trigramTransDict.items():
			for key2, value in transitions.items():
				self.trigramTransList.append([key1+':'+key2]+list(value))
		self.trigramTransListSort = sorted(self.trigramTransList, key=lambda transition: transition[3])

		# print 'Minimum trigram transition: ',self.trigramTransListSort[0]

		#* End triphone section.
		
	def wordScore(self,word,order,laplaceFactor=0):
		## Input a word (string, NOT list of blocks) and the FSM order
			# order: 1 = unigram, 2 = digram, 3 = trigram
		## Output per-block word frequency & denominator
		# Fails on illegal items.

		## Added laplaceFactor for dumbest-smoothing; default is 0, meaning that unattested transitions get frequency of 0 (i.e., no smoothing); using e.g. 0.1 means that unattested transitions get the value of the smallest transition probability / 10.; did not add this for unigram. This feature is not for serious use.
		
		## Flow layout: first, assemble blockProbs (uniBlocks, diBlocks, triBlocks); this is the list of the raw transitional probabilities of each block in order
		## Second, give the desired output, scaling and summarizing these probability sequences as specified

		blocks = []
		blockProbs = []
		# e.g. 'pildres'

		# Add the bookend nulls back
		augStr = self.start + word + self.end
		# e.g. '\x00pildres\x00'
		
		if order == 1:
			# raise Exception('wordScore() needs a valid order argument (2 or 3).')
			pass
		elif order == 2:
			transDict = self.digramTransDict
			offsetMod = 0
		elif order == 3:
			transDict = self.trigramTransDict
			offsetMod = 1
		else:
			#! Error; does this exception end the function (break)?
			raise Exception('wordScore() needs a valid order argument (1, 2, or 3).')

		# Laplace Smoothing placeholder entry; use this is replace an illegal transition
			# trigramLaplaceFactor scales the minimum transition directly; 0.1 = min_prob/10

		# trigram Laplace filler
		trigramMinTransBlock = self.trigramTransListSort[0]
		# print trigramMinTransBlock

		trigramMinTransFreq = trigramMinTransBlock[1] * laplaceFactor
		# print trigramMinTransFreq
		trigramMinTransDenominator = trigramMinTransBlock[2]
		# print trigramMinTransDenominator
		
		trigramMinRelFreq = trigramMinTransFreq/trigramMinTransDenominator
		try:
			trigramMinLogFreq = math.log(trigramMinRelFreq)
		except ValueError:
			# if you use laplaceFactor = 0, the log is undef.; NA
			trigramMinLogFreq = 'NA'
		 
		trigramMinSqrtFreq = math.sqrt(trigramMinRelFreq)

		# print trigramMinTransBlock,trigramMinTransFreq

		self.trigramLaplaceReplacement = [trigramMinTransFreq,trigramMinTransDenominator,trigramMinRelFreq,trigramMinLogFreq,trigramMinSqrtFreq]

		# digram Laplace filler
		digramMinTransBlock = self.digramTransListSort[0]
		digramMinTransFreq = digramMinTransBlock[1] * laplaceFactor
		digramMinTransDenominator = digramMinTransBlock[2]
		
		digramMinRelFreq = digramMinTransFreq/digramMinTransDenominator
		try:
			digramMinLogFreq = math.log(digramMinRelFreq)
		except ValueError:
			# if you use laplaceFactor = 0, the log is undef.; NA
			digramMinLogFreq = 'NA'
		digramMinSqrtFreq = math.sqrt(digramMinRelFreq)

		self.digramLaplaceReplacement = [digramMinTransFreq,digramMinTransDenominator,digramMinRelFreq,digramMinLogFreq,digramMinSqrtFreq]

		# also, set a flag for the whole word: 'legal', 'illegal'
		# initialize as 'legal' and change on KeyError
		legalityFlag = 'legal'

		# start with the base probability of the start block, log values

		nextBlock = augStr[0:(2+offsetMod)]
		try:
			trans = transDict['\x00'][nextBlock]
		except KeyError:
			trans = self.trigramLaplaceReplacement
			legalityFlag = 'illegal'
		
			# look up the value list for the first block (transition from null)
			# this is a tuple; 
			# '\x00{t': (10, 11382, 0.0008785802143735723, -7.037203346169024, 0.029640853806420158),
			# block trans freq Ft, transition denominators Dt, Ft/Dt, log Ft/Dt, sqrt Ft/Dt

		# initialize the blockProbs with the start block tuple; this will be a list of tuples
		blockProbs = [trans] 
		blocks = [self.start+':'+nextBlock]

		# e.g. '\x00pildres\x00'
		
		# Here, the offsetMod slices the order 3 string _shorter_, because the blocks are longer.
		# So, a 7-phone word has length 9 after the start/end blocks are added; that's 6 digrams/5 trigrams transitions.

		# print augStr
		# print len(augStr)
		# print offsetMod
		numBlocks = len(augStr)-(2+offsetMod)

		# print nextBlock
		# print blockProbs
		# print blocks
		# print numBlocks," ---\n"

		for blockOffset in xrange(numBlocks):
			# print '\n',blockOffset
			# e.g. in xrange(9-3)=6
			# slice off the current 2 blocks
			curBlock1 = augStr[blockOffset:(blockOffset+(2+offsetMod))]
			# e.g. [0:2] = '\x00p'

			curBlock2 = augStr[blockOffset+1:(blockOffset+(3+offsetMod))]
			# e.g. [1:3] = 'pi'
			
			# here, look up the info the the transition and add it to the list
			# use the try block here in order to catch illegal transitions

			try:
				blockProbs.append(transDict[curBlock1][curBlock2])
			except KeyError:
				blockProbs.append(self.trigramLaplaceReplacement)
				legalityFlag = 'illegal'

			# add the block itself (string) to the list
			blocks.append(curBlock1+':'+curBlock2)
					
		# return the list of tuples for all transitions
		blocksList = zip(blocks,blockProbs)

		rawScore = 1
		logScore = 0
		sqrtScore = 1

		# if illegal and laplaceFactor = 0, can't do a logScore
		if legalityFlag == 'illegal' and laplaceFactor == 0:
			for transBlock in blockProbs:
				rawScore *= transBlock[2]
				# logScore += transBlock[3]
				sqrtScore *= transBlock[4]
			logScore = 'NA' 
		else:
			for transBlock in blockProbs:
				rawScore *= transBlock[2]
				# print legalityFlag,laplaceFactor,transBlock
				logScore += transBlock[3]
				sqrtScore *= transBlock[4]

		return [word,order,legalityFlag,laplaceFactor,rawScore, logScore, sqrtScore]+blocksList

	def scoreWords(self, filename,laplaceFactor=0,verbose=False):
		## Outputs text file of all nonwords with their wordscores, CSV format.
			## Format: word,order,legalityFlag,laplaceFactor,rawScore,logScore,sqrtScore
			## Separate lines for each word under each order.
		## Input file should be 1 string word per line (e.g. 'wordsetS.txt') 
		## Input file must have a trailing newline, or the newline stripper will instead remove the last word character.

		f = open(filename, 'r')
		lines = f.readlines()
		f.close()
		
		templist = []
		outfile = open(filename[:-4]+'_scored.csv', 'w')
		for line in lines:
			# strip newline before scoring
			line = line[:-1]
			
			
			# If you feed bad input (modified wordlists), uniScore will fail; can wrap with Try/Except to avoid this if needed.
			# wordScore shifts the KeyError issue for order 2,3, but leaves unigrams alone; so leave this in; this should only happen if the input is bad anyway (all unigrams in the valid alphabet are attested once)

			if verbose:
				tempScore2 = self.wordScore(line,self.DIPHONE_SIZE,laplaceFactor)
				tempScore3 = self.wordScore(line,self.TRIPHONE_SIZE,laplaceFactor)
				templist.append([tempScore2,self.scoreTransform(tempScore2,numpy.min), self.scoreTransform(tempScore2,numpy.max)])
				templist.append([tempScore3,self.scoreTransform(tempScore3,numpy.min), self.scoreTransform(tempScore3,numpy.max)])
			else:
				tempScore2 = self.wordScore(line,self.DIPHONE_SIZE,laplaceFactor)[:7]
				tempScore3 = self.wordScore(line,self.TRIPHONE_SIZE,laplaceFactor)[:7]
				templist.append(tempScore2)
				templist.append(tempScore3)

		for line in templist:
			# outLine = line + 
			outfile.write(str(line)[1:-1]+'\n')
	
		outfile.close()

	def genPath(self, wordLength, order):
		'Returns a path generated by the FSM'
		'wordLength controls min/max length'
		## Used by sampleWords()

		# select a starting block
		# and initialize output string to everything after the start symbol
		# genPath is the accumulated path, a list of blocks
		
		# Loop continues until the path hits an endblock
		# If block1 is an endblock, correctly ends (returns False)
		
		# Link the order-appropriate self.transDict to the local var transDict.

		if order == 2:
			transDict = self.digramTransDict
		elif order == 3:
			transDict = self.trigramTransDict
		else:
			#! Error; does this exception end the function (break)?
			raise Exception('genPath() needs a valid order argument (2 or 3).')

		# Assign the appropriate block1set based on the order-based transDict
		block1set = transDict['\x00'].keys()
		
		tempPath = []
		
		while True:	
			# Start with a random block1
			tempPath = [random.choice(block1set)]
			# e.g. ['\x00i']
			# print tempPath

			# Loop builds until the length is correct
			while len(tempPath) < wordLength:
				# Stop and exit if word is required length
				# Otherwise, continue and add another block.
				# Find continuations
				# Skip if it hit an endblock (no continuation)
				try:
					newSet = transDict[tempPath[-1]].keys()
				except KeyError:
					break

				# Because it's possible for the trigramTransDict to contain a small number of keys with value '{}', have to manually avoid those.
				# E.g. 	'\x00$\x00': {},'\x001\x00': {},'\x002\x00': {},'\x005\x00': {},'\x00@\x00': {},'\x00i\x00': {}
				if newSet == []:
					break

				# print 'tempPath: '+str(tempPath)
				# print 'newSet: '+str(newSet)
				newBlock = random.choice(newSet)
				tempPath.append(newBlock)
		
			# If the path is complete (last block is endblock), return it

			if tempPath[-1][-1] == self.end:
				if len(tempPath) == wordLength:
					return tempPath	 
			#! Can do an AND here instead of nested ifs
			#! This check really isn't necessary; belt and suspenders.
			# If nothing gets returned (bad word), try again with a new random block.	
		
	def sampleWords(self, wordLength, order, wordCount, maxAttempts = 100000):
		## Instead of running across the full block1set/pathspace, generates (N=wordCount) random-path words.
		## Relies on genPath() function to add new words to the wordlist.

		## Returns a word list (list of strings).
		## Also saves that list to self.words.

		## wordLength : phoneme-length of the strings
		## wordCount : size of sample (number of words generated)
		## maxAttempts : limiter on the total number of iterations, in case the process is too inefficient (probably remove after testing?); default to a high but finite number


		# Initialize the master word lists, control vars
		wordList = []	# list of paths (which are lists of blocks)
		wordListS = []	# list of strings (stringed paths = words)
		blockCount = 0

		# Adjust wordLength for order=2 (because it's actually blocks and they're smaller)
		letterLength = wordLength
		if order == 2:
			wordLength += 1

		#cleanup: does this do anything
		# outLog = open("fsm_log.txt", "w")
		# outLog.close()

		# Main loop: proceed until maxAttempts or wordCount is reached.
		for i in xrange(maxAttempts):
			# If there are enough words already, exit the main loop.
			if len(wordList) >= wordCount:
				break
			
			#debug: Keep track of progress.
			#if i % 1000 == 0: print "Attempts: "+str(i)

			# If more words needed, try to find one.
			# Append a new random path (list of blocks) as a list to the wordList (list of lists)
			# Only appends if the path is valid (not 'NoneType')
			temp = self.genPath(wordLength,order)
			if temp != False:
				#debug
				#print temp
				wordList.append(temp)

		# After all word-paths found, convert to strings
		wordListS = [self.pathToString(path) for path in wordList]

		# Console output (optional)
		print "Words Generated: "+str(len(wordList))

		# Write out the wordlists (path and string formats) to files:
		if len(wordList) > 0:
			#E.g., filename: "wordset_6_sample_80.txt" is length 6, 80 items
			# filename should now include order setting
			wordFile = open("wordset_"+str(letterLength)+'_order_'+str(order)+"_sample_"+str(wordCount)+".txt", "w")
			wordFileS = open("wordsetS_"+str(letterLength)+'_order_'+str(order)+"_sample_"+str(wordCount)+".txt", "w")
			
			for word in wordList:
				wordFile.write(str(word)+"\n")
			
			for word in wordListS:	
				wordFileS.write(str(word)+"\n")

			wordFile.close()
			wordFileS.close()

		self.words = wordListS
		print len(self.words),"words also saved to self.words; new runs will overwrite."
		return wordListS

	def sampleFillers(self, wordLength, wordCount, weighted=True):
		## Generates unigram filler words, based on the unigram model from the input file.
		## Arguments: length, # to generate, frequency-weighted = True/False
		## Returns list of strings
		## Saves list of strings to file (e.g. 'fillers_4_sample_500_weighted.txt')

		## Results may be legal for diphone/tripone models, so use scoreWords() to filter for 0 values.
		## Unigram scores may be generated separately, based on weighted model.

		wordList = []

		phones, relFreqs, cumRelFreqs = zip(*self.unigramRelFreqs)
		
		fillerCount = 0
		if weighted == False:
			fillerFilename = "fillers_"+str(wordLength)+"_sample_"+str(wordCount)+"_unweighted.txt"
			while fillerCount < wordCount:
				# combine N random phones into a word
				tempWord = ""
				for j in xrange(wordLength):
					tempPhone = random.choice(phones)
					tempWord += tempPhone
				
				if '\x00' in tempWord:
					pass
				else:
					wordList.append(tempWord)
					fillerCount += 1
		
		elif weighted == True:
			fillerFilename = "fillers_"+str(wordLength)+"_sample_"+str(wordCount)+"_weighted.txt"
			while fillerCount < wordCount:
				# combine N random phones into a word
				tempWord = ""
				for j in xrange(wordLength):
					randTemp = random.random()
					indexTemp = bisect.bisect(cumRelFreqs, randTemp)
					tempWord += phones[indexTemp]

				# print tempWord
				if '\x00' in tempWord:
					pass
				else:
					wordList.append(tempWord)
					fillerCount += 1
		
		# write out list to file
		
		wordFile = open(fillerFilename, "w")
		for word in wordList:
			wordFile.write(str(word)+"\n")
		wordFile.close()
		return wordList

	def scoreTransform(self,scoreList,transformFunc):
		# scoreList is the output from wordScore()
		# scoreList[0] is the word string
		# scoreList[1] is the FSM order/block size
		# scoreList[2] is the legality
		# scoreList[3] is the laplace factor
		# scoreList[4] is the raw relative freq
		# scoreList[5] is the log relative frequency
		# scoreList[6] is the sqrt relative frequency

		# scoreList[7:] is the list of transitions
			# each transition is a tuple: (transition_string,(freq,catFreq,rawProb,logProb,sqrtProb))
				# note the nesting: e.g., transition[1][0] = freq
				# transition[1][2] = rawProb
			# the number of transitions = len(scoreList)-7

		## the sequence of rawProbs from the list of transitions is extracted to a list; rawProbList
		## the transformFunc is applied to rawProbList
		## return the output of the transformFunc
			#e.g., min, max, mean, etc.
		transitionList = scoreList[7:]
		rawProbList = []
		for transition in transitionList:
			rawProbList.append(transition[1][2])
		# print rawProbList
		# print transformFunc
		return map(transformFunc,[rawProbList])[0]







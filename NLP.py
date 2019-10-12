import PyPDF2
import textract
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from spacy_conll import Spacy2ConllParser
import spacy
spacyconll = Spacy2ConllParser()

#function for file reading and text extraction
def extractPdfText(filePath=''):

    # Open the pdf file in read binary mode.
    fileObject = open(filePath, 'rb')

    # Create a pdf reader .
    pdfFileReader = PyPDF2.PdfFileReader(fileObject)

    # Get total pdf page number.
    totalPageNumber = pdfFileReader.numPages

    # Print pdf total page number.
    print('This pdf file contains totally ' + str(totalPageNumber) + ' pages.')

    currentPageNumber = 0
    text = ''

    # Loop in all the pdf pages.
    while(currentPageNumber < totalPageNumber ):

        # Get the specified pdf page object.
        pdfPage = pdfFileReader.getPage(currentPageNumber)

        # Get pdf page text.
        text = text + pdfPage.extractText()

        # Process next page.
        currentPageNumber += 1

    if(text == ''):
        # If can not extract text then use ocr lib to extract the scanned pdf file.
        text = textract.process(filePath, method='tesseract', encoding='utf-8')
       
    return text
#text parsing
text = extractPdfText("Script.pdf")
sentences = sent_tokenize(text)
length = len(sentences)
nlp = spacy.load("en_core_web_sm")

for i in range(length):
    print(sentences[i])
    # `parse` returns a generator of the parsed sentences
    for parsed_sent in spacyconll.parse(input_str=sentences[i]):
        print(parsed_sent)
    text = sentences[i]
    doc = nlp(text)
    print("\n\n")
    print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
    print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])
    print("\n\n")
    # Find named entities, phrases and concepts
    for entity in doc.ents:
        print(entity.text, entity.label_)

FILENAME=main
FIGFILES := $(wildcard *.eps)
BIBFILES=bibliography.bib
CHAPTERS=chapters

all:

main-bibtex:
	pdflatex $(FILENAME).tex
	bibtex $(FILENAME).aux
	pdflatex $(FILENAME).tex
	pdflatex $(FILENAME).tex

main-biber:
	pdflatex $(FILENAME).tex
	biber $(FILENAME).aux
	pdflatex $(FILENAME).tex
	pdflatex $(FILENAME).tex

tar:
	tar -czvf main.tar.gz $(FIGFILES) $(FILENAME).tex

clean:
	rm -f  *.out *.blg *.log *.aux *.bbl $(CHAPTERS)/*.aux

clean-hard:
	rm -f *-eps-converted-to.pdf *.out *.blg *.log *.aux *.bbl

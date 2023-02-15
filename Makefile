FILENAME=main
FIGFILES := figs/$(wildcard *.eps)
BIBFILES=bibliography.bib
CHAPTERS=chapters
APPENDICES=appendices
LATEX = pdflatex

all: $(FILENAME).bbl
	$(LATEX) $(FILENAME).tex

$(FILENAME).bbl: $(BIBFILES)
	$(LATEX) $(FILENAME).tex
	biber $(FILENAME)
	$(LATEX) $(FILENAME).tex
	$(LATEX) $(FILENAME).tex

tar:
	tar -czvf main.tar.gz $(FILENAME).tex $(FIGFILES) $(CHAPTERS)/*.tex \
	$(APPENDICES)/*.tex $(BIBFILES) MastersDoctoralThesis.cls Makefile

clean:
	rm -f *.bcf *-blx.bib *.fdb_latexmk *.fls *.run.xml *.toc *.out \
	*.blg *.log *.aux *.bbl $(CHAPTERS)/*.aux $(APPENDICES)/*.aux

clean-hard:
	rm -f *.bcf *-blx.bib *.fdb_latexmk *.fls *.run.xml *.toc \
	*-eps-converted-to.pdf *.out *.blg *.log *.aux *.bbl $(CHAPTERS)/*.aux \
	$(APPENDICES)/*.aux

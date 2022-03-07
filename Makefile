FILENAME=main
FIGFILES := figs/$(wildcard *.eps)
BIBFILES=bibliography.bib
CHAPTERS=chapters
APPENDICES=appendices

all: $(FILENAME).bbl
	pdflatex $(FILENAME).tex

$(FILENAME).bbl: $(BIBFILES)
	pdflatex $(FILENAME).tex
	biber $(FILENAME)
	pdflatex $(FILENAME).tex
	pdflatex $(FILENAME).tex

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

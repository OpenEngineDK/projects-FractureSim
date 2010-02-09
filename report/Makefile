pdf:
	max_print_line=12000000	pdflatex -halt-on-error -output-directory=build report.tex && \
	max_print_line=12000000	pdflatex -halt-on-error -output-directory=build report.tex && \
	cd build && \
	export BIBINPUTS="..:" && \
	bibtex report.aux && \
	cd ..  &&  \
	max_print_line=12000000	pdflatex -halt-on-error -output-directory=build report.tex && \
	max_print_line=12000000	pdflatex -halt-on-error -output-directory=build report.tex

clean:
	rm -rf build
	mkdir build

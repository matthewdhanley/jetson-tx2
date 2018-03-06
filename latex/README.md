# LaTeX in Linux
1. On Ubuntu, install `texlive`
```
$ sudo apt-get update
$ sudo apt-get install texlive-full
```
This is a pretty big file. You can also usually get away with
```
$ sudo apt-get update
$ sudo apt-get install texlive texlive-base
```
but if you run into issues, just go back and install the full version.

2. Make a Tex file and name it with the `.tex` extension

3. Make a PDF
```
$ pdflatex file.tex
```

4. Check out your work by opening it with the Document Viewer
```
$ evince file.pdf
```

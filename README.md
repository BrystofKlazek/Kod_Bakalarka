V rámci pouhého prototypování jsem nedal vícekrát užívané funkce do vlastního souboru, který bych následně includnul, nýbrž jsem je pouze překopírovával. Celý kód je tedy velmi poslepován. Na funkčnost jenoho celku to však nemá žádný vliv. Scripty nejsou koncipovány tak, že jeden script - jeden běh a jeden graf. Občas jsem určitou funkčnost odkomentoval ři zakomentoval. cluster_8.csv je soubor do sir_fit.py a SEIR_fit.py, který je volán pro fit SIR/SEIR modelu na prasata na jih od Poznaně. Kód i databázi mám ve stejné directory, takže sir_fit.py resp. SEIR_fit.py volám způsobem:

python sir_fit.py cluster_8.csv -N 3500 --gamma 0.23 --I0 1 --plot --smooth 7

resp.

python SEIR_fit.py cluster_8.py -N 3500 -gamma 0.23 --sigma 0.22 --I0 2 --E0 4 --plot --smooth 7

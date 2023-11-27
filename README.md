# Facharbeit Informatik

Code des Programs, welches zu Testzwecken des überwachten bzw. unüberwachten Lernens genutzt wurde.

## Ausführen

1. Herunterladen der vorliegenden Dateien

2. Nutzen der Entwicklung [Unity]([https://unity.com/de]) to install foobar.

3. Einstellen der gewünschten Parameter in Inspektoren von:
  - Controller:
      * Learn Type gibt den Algorithmus an, welcher getestet werden soll.
      * !WIP! Activation Function ändert die Aktivierungsfunktion des neuronalen Netzwerkes. Bisher wurde allerdings nur mit TanH gearbeitet.
      * Cost Type ist das Äquivalent für die Kostenfunktion. (funktioniert alles?)
  - Secondary UINN:
      * Skipped Layer Num gibt die Anzahl der versteckten Ebenen auf dem zweiten Screen an. Hat man drei Layer und gibt zwei an, wird nur der letzte Layer dargestellt.

4. Durch Drücken auf Play wird die Simulation gestartet, die Buttons sollten selbsterklärend sein.
Der Drawmode wurde noch nicht implementiert!

## Savesystem:

Ein neuronales Netzwerk kann über "Save" gespeichert werden. 
Der Index, der zum Laden notwendig ist, wird ausgegeben. 
Es sind standardmäßig bereits Netzwerke eingespeichert,
durch eintragen der Ziffer "5" kann beispielsweise ein 
vollständig trainiertes Netzwerk geladen werden.

## Funktionsunfähig:

Die Treffsicherheit eines Clustering-Algorithmus ist schwierig zu bestimmen, wird deshalb zur Zeit noch mit Ausgabe von 1 ignoriert.

Activation Function und Cost Function wurden bereits addressiert, zuverlässig sind jedenfalls TanH und Mean Square Error.

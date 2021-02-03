# Plate-Counter

Modifiziertes U-Net zur semantischen Segmentierung

Programmiersprache Python. Die installierten Bibliotheken befinden sich in der Datei „requirements.txt“.

Voraussetzung: Trainingsbilder, dazugehörende Annotationsmasken, Validierungsbilder und Testbilder.
Ablauf

Die Bilder und Masken in entsprechende Ordner „images_train“ (Bilder) bzw. „binary_masks“ (Annotationsmasken) kopieren

Augmentierte Trainingsbilder erzeugen mit Skript data_augmentation.py

Die augmentierten Bilder und Masken werden zusammen mit den Originalbildern und Masken in dieUnterordner „img“ und „masks“ des Ordners „images_augmented“ gespeichert

Die Skripte „make_trainfolders.py“, „make_validationfolders.py“ und „make_testfolders.py“ erzeugen die Ordnerstrukturen für die Trainings- und Validierungsbilder und -masken und die Testbilder. Die Testbilder (dürfen nicht in den Trainingsbildern vorhanden sein) in den Testbildordner „images_test“ kopieren

Modell trainieren mit Skript „train_model.py“ (Modell wird gespeichert). Oder Vorhersagen mit bereits trainiertem Modell mit Skript „predict.py“ (lädt zuvor gespeichertes Modell)

Die Ausgabe der Ergebnisbilder (Probabilty maps und Binärbilder) erfolgt im Ordner „results“

Auswerten der Ergebnisbilder (Anzahl der Objekte, Positionen und Größe bestimmen) mit Skript „evaluate_results.py“.

Die Ergebnisse werden in jeweils einer JSON-Datei pro Bild gespeichert. Zusätzlich wird die Anzahl der Objekte in eine CSV-Datei im Ordner „CSV_lists“ gespeichert. Die Visualisierung der Ergebnisse erfolgt mit Bounding Boxes


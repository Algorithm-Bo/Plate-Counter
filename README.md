# Plate-Counter
Modifiziertes U-Net zur semantischen Segmentierung
Voraussetzung: Trainingsbilder und dazugehörende Annotationsmasken

Programmiersprache Python.
Die installierten Bibliotheken befinden sich in der Datei „requirements.txt“

Bilder und Masken in entsprechende Ordner „images_train“ (Bilder) bzw. „binary_masks“ (Annotationsmasken) kopieren
Skript „make_trainfolders.py“ aufrufen (Erzeugt Ordnerstruktur mit Trainingsdatensatz)

Augmentierte Bilder erzeugen mit Skript data_augmentation.py
Namen von ab Nummer nach Originalbilder.PNG vergeben
Orginaltrainingsbilder und Masken zu den Unterordnern „img“ und „masks“ hinzufügen
Skript „make_trainfolders.py“ aufrufen (Erzeugt Ordnerstruktur inklusive augmentierter Bildern)

Ordner der Originalbilder und augmentierten Bilder in gemeinsamen Ordner kopieren (Trainingsdatensatz fürs FCN)

Testbilder (dürfen nicht in den Trainingsbildern vorhanden sein) in den Testbildordner „images_test“ kopieren
Skript „make_testfolders.py“ aufrufen (Erzeugt Ordnerstruktur mit Testbildern)

Modell trainieren mit Skript „train_model.py“ (Modell wird gespeichert unter „my_model.h5“)
Oder Vorhersagen mit vortrainiertem Modell „predict.py“ (lädt „my_model.h5)
Ausgabe der Ergebnisbilder (Binärbilder) jeweils in Ordner „results“

Auswerten der Ergebnisbilder (Zählen+Positionen) mit Skript „evaluate.py“
Ausgabe der Zählergebnisse des Testdatensatzes in CSV Datei „results.CSV“ in Ordner „result_lists_CSV“
Ausgabe der Ergebnisse (Objektpositionen im Einzelbild) in JSON Dateien „Nummer_Testbild.JSON“ im Ordner „JSON_counted_objects“

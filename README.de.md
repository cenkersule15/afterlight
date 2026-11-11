# AFTERLIGHT — Schadensanalyse mit Satellitenbildern

Ein dreisprachiger Forschungsarbeitsbereich von Cenker Sule: echte Bilddaten, trainierte Modelle und nachvollziehbare Auswertung.

[English](README.md) · [Türkçe](README.tr.md)

## Studio 02: überarbeitete Oberfläche

Dunkles/helles Farbschema, durchsuchbare Bildbibliothek, im Browser gespeicherte Markierungen, Vergleich nebeneinander, Fokusmodus, Bereichsfilter nach Klasse und Fläche sowie szenenspezifische IoU/Dice/F1. Cmd/Ctrl+K öffnet die Bibliothek; links/rechts wechselt Szenen, R startet die Analyse, L schaltet Referenzlabels, F den Fokusmodus. Modellgewichte, Datenteilung und Gesamtscores bleiben unverändert.


## Start mit VS Code und Docker

ZIP entpacken und den Ordner **afterlight** mit `compose.yaml` in VS Code öffnen. Docker Desktop starten. Im Terminal:

```bash
docker compose up --build
```

Oberfläche: **http://localhost:8080**. API-Dokumentation: **http://localhost:8080/api/docs**.
Der erste Build lädt Abhängigkeiten und kann mehrere Minuten dauern. Auf dem Host sind bei Docker-Nutzung weder Node noch Python erforderlich. Mit Ctrl+C und `docker compose down` stoppen.

## Funktionen

- 66 echte xBD-Vorher/Nachher-Bildpaare aus 9 Katastrophenereignissen.
- Englische, türkische und deutsche Oberfläche.
- Bildvergleich, Zoom, Referenzannotation und getrennte Modellvorhersagen.
- PyTorch Pair U-Net und scikit-learn Random Forest mit gespeicherten Gewichten.
- Schadensoverlay, Untersuchung zusammenhängender Bereiche und JSON-Export.
- Datenlabor, Klassenverteilung, Lernkurve, Konfusionsmatrix und gemessene Testmetriken.
- Upload ausgerichteter PNG/JPEG-Paare mit gleichen Abmessungen, maximal 8 MiB / 16 MP pro Bild. Keine erfundenen Genauigkeitswerte für unbeschriftete Uploads.

## Tatsächlicher Forschungsstand

Dieser Stand ist ein **lauffähiger kleiner Forschungspilot**, kein leistungsstarkes produktives Schadenserkennungssystem. Schaden-Makro-F1: ungefähr **0,103** für Pair U-Net und **0,111** für Random Forest. Die niedrigen Werte verlangen mehr Daten und weitere Modellarbeit. Keine Nutzung für Gebäude- oder Einsatzsicherheitsentscheidungen.

Ereignisgetrennte Aufteilung: 37 Trainings-, 13 Validierungs- und 16 Testpaare. Ein Ereignis bleibt vollständig in einer Teilmenge. Keine offizielle xView2-Wettbewerbsauswertung. Bilder liegen mit 512 px vor, Modelle werden bei 128 px ausgewertet. Das sechskanalige U-Net segmentiert Hintergrund und vier Schadensklassen gemeinsam; keine zwei unabhängig trainierten Stufen. Zusammenhängende Bereiche sind keine bestätigten einzelnen Gebäude.

## Separat trainieren

```bash
docker compose --profile training run --rm train
docker compose restart api
```

40 Epochen / 128 px erneuern die gespeicherten Modelle. Beim normalen Start wird nicht neu trainiert. `data/` und `artifacts/` bleiben als lokale Bind-Mounts erhalten. Docker nutzt CPU; Apple MPS ist in Mac-Containern nicht verfügbar.

## Daten und Lizenz

xBD / xView2: Gupta et al., DIU / CMU SEI; Bilddaten: Maxar Open Data Program. Herkunft, festgelegte Revision, Prüfsumme und Transformationen stehen in `data/catalog.json`. Anwendungscode: MIT. Gebündelte Daten und Pilotmodell-Artefakte: **CC BY-NC-SA 4.0** mit Namensnennung, nichtkommerzieller Nutzung und Share-Alike-Bedingungen.

Quelle: https://huggingface.co/datasets/kshitijrajsharma/xview2-xbd

## Prüfung

14 Python-Tests, TypeScript-Prüfung und Frontend-Build bestanden. CPU-Training und Inferenz wurden ausgeführt. Docker ist in der Autorenumgebung nicht verfügbar; Container-Build und Start müssen auf einem Docker-Host abschließend geprüft werden.

Architektur, Import größerer xBD-Daten, Wiederherstellung der Daten nach einem Source-only-GitHub-Clone und weitere Einschränkungen: [English README](README.md).

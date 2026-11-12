# Sitzung: WP-Lastprognose und Vorlaufregelung

Datum: 12. November 2026

## Ausgangsfrage

Der berechnete Vorlauf liegt deutlich über dem Rücklauf; die Wärmepumpe (WP)
könnte deshalb bald starten. Trotzdem zeigt `sensor.ml_heating_wp_load_forecast`
für mehrere Stunden nur Standby-Verbrauch. Anschließend wurde diskutiert, ob die
bei PV-Überschuss ausgelassene Overshoot-Korrektur in der Lastprognose
berücksichtigt werden sollte und wie sich die tatsächliche Regelstrategie
einfacher oder genauer abbilden ließe.

## Beobachtetes Beispiel

### Screenshot der WP

| Größe | Wert |
|---|---:|
| Außentemperatur | 9,5 °C |
| Betriebsmodus | aus |
| Verdichterfrequenz | 0 Hz |
| Berechneter Vorlauf Heizung | 26,3 °C |
| Berechneter Vorlauf Kühlung | 18,6 °C |
| Tatsächlicher Vorlauf | 23,4 °C |
| Rücklauf | 23,6 °C |

Der Screenshot allein belegt keinen bevorstehenden Start: Dieser hängt von der
internen WP-Regelung ab.

### Lastprognose von 09:21 Uhr

Die Prognose wurde am 12.11.2026 um `08:21:08 UTC` bzw. `09:21:08 +01:00`
erzeugt, also etwa zehn Minuten vor dem unten zitierten Regelungslog. Der
Sensorzustand betrug `0,0190 kWh`; `horizon_hours` war 48.

Die folgenden Stundenwerte stammen aus dem bereitgestellten Sensor. Alle
Zeitpunkte sind lokale Startzeiten mit Offset `+01:00`; sämtliche Werte sind
kWh je Stunde. `Gesamt` entspricht auch `forecast[].value`. Die
Warmwasser-Prognose war in allen 48 Stunden null.

| Start | Heizung thermisch | Heizung elektrisch | Standby elektrisch | Gesamt elektrisch | Baseline thermisch |
|---|---:|---:|---:|---:|---:|
| 12.11. 09:00 | 0 | 0 | 0,019 | 0,019 | 0 |
| 12.11. 10:00 | 0 | 0 | 0,05 | 0,05 | 0,3104 |
| 12.11. 11:00 | 0 | 0 | 0,05 | 0,05 | 0,1959 |
| 12.11. 12:00 | 0 | 0 | 0,05 | 0,05 | 0,175 |
| 12.11. 13:00 | 0 | 0 | 0,05 | 0,05 | 0,3369 |
| 12.11. 14:00 | 0 | 0 | 0,05 | 0,05 | 0,632 |
| 12.11. 15:00 | 0 | 0 | 0,05 | 0,05 | 0,991 |
| 12.11. 16:00 | 3,1812 | 0,483 | 0,05 | 0,533 | 1,2054 |
| 12.11. 17:00 | 3,1812 | 0,5051 | 0,05 | 0,5551 | 1,4163 |
| 12.11. 18:00 | 3,1812 | 0,5434 | 0,05 | 0,5934 | 1,6814 |
| 12.11. 19:00 | 3,1812 | 0,5856 | 0,05 | 0,6356 | 1,9295 |
| 12.11. 20:00 | 3,1812 | 0,6003 | 0,05 | 0,6503 | 2,01 |
| 12.11. 21:00 | 3,1812 | 0,6079 | 0,05 | 0,6579 | 2,0518 |
| 12.11. 22:00 | 2,3454 | 0,454 | 0,05 | 0,504 | 2,0914 |
| 12.11. 23:00 | 2,3454 | 0,4599 | 0,05 | 0,5099 | 2,1323 |
| 13.11. 00:00 | 2,3454 | 0,4722 | 0,05 | 0,5222 | 2,202 |
| 13.11. 01:00 | 2,3454 | 0,492 | 0,05 | 0,542 | 2,3191 |
| 13.11. 02:00 | 2,3454 | 0,5086 | 0,05 | 0,5586 | 2,4109 |
| 13.11. 03:00 | 2,3454 | 0,5263 | 0,05 | 0,5763 | 2,5031 |
| 13.11. 04:00 | 2,9715 | 0,6874 | 0,05 | 0,7374 | 2,581 |
| 13.11. 05:00 | 2,9715 | 0,6945 | 0,05 | 0,7445 | 2,6064 |
| 13.11. 06:00 | 2,9715 | 0,6981 | 0,05 | 0,7481 | 2,6169 |
| 13.11. 07:00 | 2,9715 | 0,6874 | 0,05 | 0,7374 | 2,5427 |
| 13.11. 08:00 | 2,9715 | 0,6735 | 0,05 | 0,7235 | 2,0243 |
| 13.11. 09:00 | 2,9715 | 0,657 | 0,05 | 0,707 | 1,2877 |
| 13.11. 10:00 | 0 | 0 | 0,05 | 0,05 | 0,8421 |
| 13.11. 11:00 | 0 | 0 | 0,05 | 0,05 | 0,6267 |
| 13.11. 12:00 | 0 | 0 | 0,05 | 0,05 | 0,5817 |
| 13.11. 13:00 | 0 | 0 | 0,05 | 0,05 | 0,6848 |
| 13.11. 14:00 | 0 | 0 | 0,05 | 0,05 | 1,0015 |
| 13.11. 15:00 | 0 | 0 | 0,05 | 0,05 | 1,5152 |
| 13.11. 16:00 | 4 | 0,7743 | 0,05 | 0,8243 | 1,9856 |
| 13.11. 17:00 | 4 | 0,8163 | 0,05 | 0,8663 | 2,2302 |
| 13.11. 18:00 | 4 | 0,851 | 0,05 | 0,901 | 2,3595 |
| 13.11. 19:00 | 4 | 0,8844 | 0,05 | 0,9344 | 2,4674 |
| 13.11. 20:00 | 4 | 0,9067 | 0,05 | 0,9567 | 2,5384 |
| 13.11. 21:00 | 4 | 0,9159 | 0,05 | 0,9659 | 2,5706 |
| 13.11. 22:00 | 2,6667 | 0,6106 | 0,05 | 0,6606 | 2,5725 |
| 13.11. 23:00 | 2,6667 | 0,6106 | 0,05 | 0,6606 | 2,5765 |
| 14.11. 00:00 | 2,6667 | 0,6265 | 0,05 | 0,6765 | 2,6266 |
| 14.11. 01:00 | 2,6667 | 0,6467 | 0,05 | 0,6967 | 2,7057 |
| 14.11. 02:00 | 2,6667 | 0,6433 | 0,05 | 0,6933 | 2,698 |
| 14.11. 03:00 | 2,6667 | 0,6502 | 0,05 | 0,7002 | 2,7312 |
| 14.11. 04:00 | 2 | 0,4957 | 0,05 | 0,5457 | 2,7709 |
| 14.11. 05:00 | 2 | 0,4957 | 0,05 | 0,5457 | 2,7709 |
| 14.11. 06:00 | 2 | 0,4984 | 0,05 | 0,5484 | 2,7839 |
| 14.11. 07:00 | 2 | 0,4984 | 0,05 | 0,5484 | 2,7709 |
| 14.11. 08:00 | 2 | 0,5069 | 0,05 | 0,5569 | 2,8098 |

Weitere Attribute des Beispiels:

- `includes`: Heizung und Standby; `climate_mode_assumed: heating`,
  `cooling_forecast: false`; Kamin aus, keine Kaminszenarien.
- COP aus gelernten Energiezählern, kalibriert: `intercept=4,2565000495`,
  `slope=0,2218730352`, 8 Stichproben, Temperaturbereich
  `0,443–7,375 °C`, beobachteter COP-Bereich `4,512–6,015`.
- `defrost_handling: net`, `defrost_mode: power_history`,
  `defrost_el_in_heating_counter: true` (konfiguriert); Stromanteil
  `0,09973`, Wärmeanteil `0,06814`, 39 historische Ereignisse, jeweils
  Blend-Faktor 0.
- Standby nicht aus Tagen gelernt (`standby_learned_days: 0`),
  angenommener Mittelwert `0,05 kW`. Warmwasser war nicht im Gesamtwert
  enthalten (`dhw_learned_days: 0`, Median-Fallback `1,1 kWh/Tag`).
- `current_hour_estimate: counter_anchor`, Kalibrierung aus Home Assistant
  über `109,465 h`, `degraded_reasons: []`.
- Die ersten 15 Preise waren `actual`, danach 33 Stunden `reused_today`.
  Ab 14.11. 00:00 fehlte die PV-Prognose (`missing_zero`, `padded: true`);
  die Wetterwerte waren für zukünftige Stunden als `weather_forecast`
  markiert. Zu Beginn und für Teile des späteren Horizonts wurde der COP
  außerhalb seines gelernten Temperaturbereichs extrapoliert.

### Regelungslog von 09:31 Uhr

- Raumtemperatur `22,72 °C`, Raumsoll `22,6 °C`, durch Kostenmodus angepasst
  auf `22,7 °C` (`+0,1 °C`).
- WP aus: gemessenes `delta_t=-0,08 °C`; für die Vorlaufsuche wurde ein
  WP-ein-ΔT von `2,61 °C` simuliert. Rücklauf im Log `23,5 °C`, tatsächlicher
  Vorlauf rund `23,42 °C`, Durchfluss `14 l/min`, Leistungsaufnahme `19 W`.
- Die Binärsuche fand `26,2 °C` bei einer 10-Stunden-Endprognose von
  `22,71 °C`; nach Smart Rounding wurde `26,0 °C` als
  `sensor.ml_vorlauftemperatur` ausgegeben. Der Screenshot mit 26,3 °C ist
  eine separate Beobachtung und nicht ohne Weiteres derselbe wirksame Sollwert.
- Für den aktuellen Zyklus wurden `22,71 °C` prognostiziert; die Trajektorie
  erreichte etwa `23,08 °C` nach 4,5–4,7 Stunden und endete bei `22,71 °C`
  nach 10 Stunden.
- Kostenmodus: Netzpreis `0,448 €/kWh`, effektiver Preis jetzt
  `0,079 €/kWh`, PV-Anteil `100 %`, angenommene Last `1,10 kW`
  (`0,40 kW` WP + `0,70 kW` Haushalt, Fallback).
- Das Log meldete: `Ignoring minor future overshoot ... immediate cycle is
  safe` und danach `Skipping overshoot correction (cost mode: cheap and PV
  covers HP demand); no undershoot predicted`.

## Diagnose des bestehenden Forecasts

Die Lastprognose in [`src/heat_forecast.py`](../src/heat_forecast.py) ist eine
eigenständige Zwei-Knoten-Simulation von Raum und Boden. Sie verwendet unter
anderem Außentemperatur, solare und andere Gewinne, geschätzte Kapazitäten,
Raumsollwerte einschließlich Preis-/PV-Offsets und den Anfangszustand des
Bodens. Sie simuliert **nicht** direkt den berechneten Vorlauf-Sollwert oder
die Start-/Stoppregelung des Verdichters.

In jedem Simulationsschritt wird nur so viel Wärme angefordert, wie nötig ist,
um die untere Komfortgrenze `Raumsoll − 0,1 °C` zu halten. Bleibt der Raum
ohne WP darüber, ergibt sich null Heizenergie. Daher können gespeicherte
Bodenwärme und solare Gewinne zu einer Standby-Prognose führen, obwohl die reale
WP aufgrund ihrer eigenen Regelung starten könnte. Welcher Eingang in diesem
konkreten Forecast ausschlaggebend war, ist aus dem Ausschnitt nicht sicher
bestimmbar.

`forecast_baseline_th_kwh` ist eine separate stationäre Bedarfsabschätzung;
eine positive Baseline bedeutet nicht, dass die dynamische Simulation im
betreffenden Zeitfenster Heizenergie anfordert.

Die Prognose glättet außerdem isolierte Spitzen innerhalb von
Sechsstundenblöcken, wobei deren Energiesumme erhalten bleibt. Die identischen
Werte von 16 bis 21 Uhr sind ein Hinweis auf diese Glättung. **16 Uhr ist
deshalb nicht als exakter prognostizierter Startzeitpunkt zu verstehen.**

Für die aktuelle Stunde kombiniert der Forecast bereits angefallene
Zählerenergie mit einer Restzeitschätzung. Bei verfügbarer aktueller Leistung
wird der elektrische Rest anhand dieser Leistung hochgerechnet; bei aktuell
stillstehender WP kann ein bevorstehender Start dabei fehlen. Der gezeigte
`counter_anchor` bedeutet nicht, dass der zukünftige Start bekannt ist.

## Bedeutung der ausgelassenen Overshoot-Korrektur

Preis-/PV-abhängige Raumziel-Offsets werden in der Lastprognose bereits
berücksichtigt. Die Entscheidung der Vorlaufregelung, eine Overshoot-Korrektur
auszulassen, wird dagegen nicht direkt in der Simulation der WP-Leistung
nachgebildet.

Im gezeigten Zyklus wurde der zukünftige Überschwinger im Log bereits als
tolerierbar bewertet und der nächste Zyklus als sicher eingestuft. Die
anschließende Skip-Meldung beweist daher **nicht**, dass der Vorlauf ohne die
Skip-Regel tatsächlich abgesenkt worden wäre.

Ein pauschaler Verbrauchsaufschlag allein aufgrund dieser Meldung wäre
unzuverlässig: Die Meldung ist kein Startsignal, und zusätzliche Wärme könnte
doppelt gezählt werden. Entscheidend ist, ob der tatsächlich wirksame
Vorlauf-Sollwert das Verhalten der WP verändert.

## Diskutierte Ansätze

### A. Rücklauf und Mindest-ΔT als Indikator

Ein einfacher vorzeichenrichtiger Vergleich wäre:

```text
Heizen: Vorlauf-Soll − Rücklauf >= ΔT-Schwelle
Kühlen: Rücklauf − Vorlauf-Soll >= ΔT-Schwelle
```

Beispiel aus dem Log: `26,0 − 23,5 = 2,5 °C`; das gelernte
`delta_t_floor` ist rund `2,61 °C`. Mit genau dieser Schwelle würde der
Indikator noch nicht auslösen. Mit den getrennten Screenshot-Werten
`26,3 − 23,6 = 2,7 °C` läge er darüber. Die Beispiele zeigen die Empfindlichkeit
gegenüber Sollwert, Rundung und Messzeitpunkt.

`delta_t_floor` beschreibt ein gelerntes Betriebs-ΔT bei laufender WP; es ist
keine nachgewiesene Startschwelle. Die Regelung nutzt es bei ausgeschalteter WP,
um für die Vorlaufsuche einen laufenden Zustand zu simulieren. Der Vergleich
eignet sich daher als Indikator, nicht als sichere Startaussage. Das tatsächlich
an der WP wirksame Soll muss bekannt sein.

### B. Start-/Stoppmodell mit Hysterese

Getrennte Start- und Stoppschwellen auf Basis geeigneter Soll-/Ist-Abweichungen
verwenden, ergänzt um Mindestlauf- und -pausenzeiten. Anfangs können diese
Parameter konfiguriert und später aus tatsächlich beobachteten WP-Zyklen
kalibriert werden. Dies bildet Takten besser ab als ein einzelner ΔT-Vergleich,
bleibt aber eine Näherung der gegebenenfalls komplexeren internen WP-Regelung.

### C. Leistung während eines erwarteten Betriebs

Bei erwartetem Betrieb lässt sich thermische Leistung näherungsweise aus
Durchfluss und Betriebs-ΔT schätzen:

```text
thermische Leistung [kW] ≈ Durchfluss [l/min] / 60
                          × 4,18 [kJ/(kg·K)] × ΔT [K]
elektrische Heizenergie ≈ thermische Leistung × Laufzeit / COP
```

Beispiel: `14 / 60 × 4,18 × 2,61 ≈ 2,55 kW` thermisch. Durchfluss und
ΔT können schwanken; Modulation, Anlauf und Abtauung müssen bei der
Interpretation berücksichtigt werden. Der COP stammt bereits aus
Energiezählern; ein zweiter Abtauaufschlag darf bei Netto-COP nicht
unbesehen hinzugefügt werden.

### D. Startwahrscheinlichkeit aus Historie

Tatsächliche Starts und Stopps mit wirksamem Sollwert, Vorlauf, Rücklauf,
Außentemperatur und WP-Zustand verknüpfen. Daraus kann statt eines harten
Schalters eine erwartete Startwahrscheinlichkeit oder Laufzeit gelernt werden.
Dies passt sich der konkreten Anlage an, benötigt aber ausreichend geeignete
Messdaten und eine Unterscheidung von Raumheizung, Warmwasser und Abtauung.

### E. Regelstrategie im Forecast nachspielen

Für jeden zukünftigen Simulationsschritt den erwarteten Vorlauf-Sollwert aus
derselben Regelstrategie ableiten, inklusive Preis-/PV-Offsets und
Overshoot-Entscheidung, und anschließend ein WP-Start-/Stopp- und
Leistungsmodell darauf anwenden. Dadurch könnte gezieltes Vorheizen sichtbar
werden. Dieser Ansatz ist aufwendiger und über lange Horizonte unsicher, da
künftige Messwerte, interne WP-Zustände und Regelentscheidungen unbekannt sind.

## Empfohlene schrittweise Umsetzung

1. Kurzfristig den tatsächlich wirksamen Vorlauf-Sollwert, Rücklauf,
   WP-Zustand sowie separate, kalibrierbare Start-/Stoppschwellen verwenden.
2. Erwartete Laufzeit und Leistung schätzen und die Wärme **als WP-Eintrag in
   die bestehende Raum-/Bodensimulation** einspeisen. Die gespeicherte Wärme
   verändert dann den späteren Heizbedarf; kein pauschales Addieren zum
   bisherigen Heizbedarf.
3. Die Restprognose der aktuellen Stunde so erweitern, dass ein erwarteter
   Start trotz momentan geringer Leistung nicht überschrieben wird. Bereits
   gemessene Energie und die simulierte Restzeit nur einmal zählen.
4. Zunächst im Vergleichs- oder Schattenbetrieb gegen historische Starts und
   gemessene Stundenenergie prüfen; insbesondere Fehlstarts, verpasste Starts
   und zeitliche Verschiebungen betrachten. Erst danach die längerfristige
   Modellierung ausbauen.

Eine konstante Fortsetzung des heutigen Vorlauf-Sollwerts über 48 Stunden wäre
keine verlässliche Annahme. Der bestehende Lastforecast prognostiziert derzeit
keine Kühlung; eine vorzeichenrichtige Kühlregel wäre eine separate
Erweiterung.

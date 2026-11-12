# Anleitung: Sensor "WP-Start- und Laufzeitschätzung" in Home Assistant bauen

Ergebnis: Sensoren, die herstellerunabhängig melden, **in wie vielen Minuten die WP startet** und **wie lange sie voraussichtlich läuft** (Heizen und Kühlen). Das Add-on liest später nur das Ergebnis.

**Zwei unabhängige Bauvarianten, je Betriebsart genau eine wählen:**
- **Variante A: Gradminuten (NIBE und ähnliche).** Schritte 3 bis 5.
- **Variante B: reine Hysterese-WP** (Start unter Soll − Hysterese, Stopp über Soll + Hysterese). Abschnitt 5b.

**Unabhängigkeit:** Die Sensoren sind reine Zusatzinformation. Das Add-on funktioniert ohne sie genau wie bisher. Heizen und Kühlen sind getrennt, jeder Modus darf eine andere Variante nutzen oder gar keinen Sensor haben. Die Sensoren verwenden **nur Entitäten deiner WP, keine Entitäten des Add-ons**; sie laufen also auch ganz ohne Add-on. Jeder Sensor meldet sein Verfahren im Attribut `method` (`degree_minutes` bzw. `hysteresis`), damit das Add-on ihn richtig einordnet.

## 1. Die Rechnung

Gradminuten (GM) summieren pro Minute die Abweichung `Vorlauf_ist − Vorlauf_soll` auf.

**Heizen**
| Zustand | Formel | Dein Beispiel |
|---|---|---|
| WP aus, Soll > Ist | `t_start = (GM − GM_start) / (Soll − Ist)` | (0 − (−200)) / (26 − 24) = **100 min** |
| WP läuft, Ist > Soll | `t_lauf = (GM_stop − GM) / (Ist − Soll)` | (0 − (−250)) / (27 − 26) = **250 min** |
| Ist ≈ Soll | unbegrenzt (WP regelt Leistung hoch) | Laufzeit wird auf Maximalwert gedeckelt |

**Kühlen** (gespiegelt, **Vorzeichen bitte an deiner Historie prüfen**, siehe Schritt 6):
| Zustand | Formel |
|---|---|
| WP aus, Ist > Soll | `t_start = (GM_start_kühlen − GM) / (Ist − Soll)` |
| WP läuft, Ist < Soll | `t_lauf = (GM − GM_stop_kühlen) / (Soll − Ist)` |

Das ist eine **lineare Schätzung** bei konstantem Ist und Soll. Beides ändert sich real (Ist driftet bei WP aus zur Rücklauftemperatur, Soll folgt Außentemperatur und der Regelung der App). Deshalb wird der Wert jede Minute neu berechnet.

## 2. Entitäten heraussuchen

Unter *Entwicklerwerkzeuge → Zustände* die passenden Entitäten suchen und unten die Platzhalter ersetzen. Die Namen hängen von deiner NIBE-Anbindung ab.

| Platzhalter im Code | Was du suchst |
|---|---|
| `sensor.nibe_degree_minutes` | Gradminuten (GM) |
| `sensor.nibe_bt2_supply_temp` | **tatsächlicher** Vorlauf (BT2/BT25, je nach Anlage) |
| `sensor.nibe_calculated_supply_temp` | **berechneter** Soll-Vorlauf Heizen (dein Screenshot: "Berechneter Vorlauf Heizung") |
| `sensor.nibe_calculated_supply_temp_cooling` | berechneter Soll-Vorlauf Kühlen |
| `sensor.nibe_compressor_frequency` | Verdichterfrequenz in Hz (> 0 = läuft) |
| `binary_sensor.warmwassermodus` | Warmwasser aktiv (hast du bereits) |
| `binary_sensor.defrost_wp` | Abtauen aktiv (hast du bereits) |
| `sensor.nibe_operating_mode` | Entität, die den aktiven Betrieb der WP zeigt (Heizen / Kühlen / Warmwasser). Den Vergleichswert in `mode_ok` an deine Werte anpassen. Gibt es so eine Entität nicht, setze `mode_ok: "{{ true }}"` und lass nur den Block aktiv, der zur Jahreszeit passt. |

Nimm als Soll den **von der WP berechneten** Vorlauf, nicht `sensor.ml_vorlauftemperatur`. Die App-Ausgabe läuft erst über deine Heizkurven-Automation, und die WP entscheidet anhand ihres eigenen Werts.

## 3. Schritt 1: Helfer für Schwellen anlegen

Die Start-/Stopp-Schwellen sind Einstellungen der WP und werden meist nicht als Sensor angeboten. Lege sie als Helfer an (*Einstellungen → Geräte & Dienste → Helfer → Zahl*) oder in YAML:

```yaml
input_number:
  wp_gm_start_heizen:
    name: WP GM Start Heizen
    min: -2000
    max: 0
    step: 10
    mode: box
    unit_of_measurement: GM
  wp_gm_stop_heizen:
    name: WP GM Stopp Heizen
    min: -500
    max: 500
    step: 10
    mode: box
    unit_of_measurement: GM
  wp_gm_start_kuehlen:
    name: WP GM Start Kühlen
    min: 0
    max: 2000
    step: 10
    mode: box
    unit_of_measurement: GM
  wp_gm_stop_kuehlen:
    name: WP GM Stopp Kühlen
    min: -500
    max: 500
    step: 10
    mode: box
    unit_of_measurement: GM
  wp_runtime_cap_min:
    name: WP Laufzeit Deckel
    min: 60
    max: 1440
    step: 30
    mode: box
    unit_of_measurement: min
```

Nach dem Neuladen die Werte setzen: **Start Heizen = −200** (dein Wert), Stopp Heizen = 0 (oder dein Wert), Deckel = 720. Kühlen-Werte aus den WP-Einstellungen übernehmen. Ohne `initial:` behält HA die Werte über Neustarts.

## 4. Schritt 2: Template-Sensor Heizen

In `configuration.yaml` (oder `packages/wp_start_estimate.yaml`) unter `template:` einfügen. Der Block ist **trigger-basiert**: Er rechnet jede Minute und bei jeder Änderung von GM, Vorlauf oder Verdichter neu. `variables` gelten für alle Sensoren im Block, damit die Logik nur einmal steht (aktuelle HA-Version nötig).

```yaml
template:
  - trigger:
      - platform: time_pattern
        minutes: "/1"
      - platform: state
        entity_id:
          - sensor.nibe_degree_minutes
          - sensor.nibe_bt2_supply_temp
          - sensor.nibe_calculated_supply_temp
          - sensor.nibe_compressor_frequency
          - binary_sensor.warmwassermodus
          - binary_sensor.defrost_wp
          - sensor.nibe_operating_mode
    variables:
      mode_ok: "{{ is_state('sensor.nibe_operating_mode', 'heating') }}"
      data_ok: >-
        {{ is_number(states('sensor.nibe_degree_minutes'))
           and is_number(states('sensor.nibe_bt2_supply_temp'))
           and is_number(states('sensor.nibe_calculated_supply_temp')) }}
      gm: "{{ states('sensor.nibe_degree_minutes') | float(0) }}"
      sup: "{{ states('sensor.nibe_bt2_supply_temp') | float(0) }}"
      req: "{{ states('sensor.nibe_calculated_supply_temp') | float(0) }}"
      running: "{{ states('sensor.nibe_compressor_frequency') | float(0) > 0 }}"
      blocked: >-
        {{ is_state('binary_sensor.warmwassermodus', 'on')
           or is_state('binary_sensor.defrost_wp', 'on') }}
      gm_start: "{{ states('input_number.wp_gm_start_heizen') | float(-200) }}"
      gm_stop: "{{ states('input_number.wp_gm_stop_heizen') | float(0) }}"
      cap: "{{ states('input_number.wp_runtime_cap_min') | float(720) }}"
      need: "{{ req - sup }}"
    sensor:
      - name: "WP Minuten bis Start Heizen"
        unique_id: wp_minutes_to_start_heating
        unit_of_measurement: "min"
        state_class: measurement
        state: >-
          {% if not (mode_ok and data_ok) or blocked %}
            {{ none }}
          {% elif running %}
            0
          {% elif need > 0.05 %}
            {{ [((gm - gm_start) / need) | round(0), 0] | max | int }}
          {% else %}
            {{ none }}
          {% endif %}
        attributes:
          status: >-
            {% if not (mode_ok and data_ok) %}unknown
            {% elif blocked %}blocked
            {% elif running %}running
            {% elif need > 0.05 and ((gm - gm_start) / need) <= 30 %}starting_soon
            {% else %}idle{% endif %}
          mode: heating
          method: degree_minutes
          expected_runtime_min: >-
            {% if not (mode_ok and data_ok) or blocked or not running %}
              {{ none }}
            {% elif (sup - req) > 0.05 %}
              {{ [[((gm_stop - gm) / (sup - req)) | round(0), 0] | max, cap] | min | int }}
            {% else %}
              {{ cap | int }}
            {% endif %}
          runtime_unbounded: >-
            {{ running and (sup - req) <= 0.05 }}
          confidence: >-
            {% if not (mode_ok and data_ok) or blocked %}0
            {% elif running %}0.8
            {% else %}0.6{% endif %}
          gm_now: "{{ gm }}"
          gm_start: "{{ gm_start }}"
          gm_stop: "{{ gm_stop }}"
          supply_actual_c: "{{ sup }}"
          required_supply_c: "{{ req }}"
          updated_at: "{{ now().isoformat() }}"
      - name: "WP Laufzeit Heizen"
        unique_id: wp_expected_runtime_heating
        unit_of_measurement: "min"
        state_class: measurement
        state: >-
          {% if not (mode_ok and data_ok) or blocked or not running %}
            {{ none }}
          {% elif (sup - req) > 0.05 %}
            {{ [[((gm_stop - gm) / (sup - req)) | round(0), 0] | max, cap] | min | int }}
          {% else %}
            {{ cap | int }}
          {% endif %}
```

**Wichtig zu den Zuständen:**
- `0` bedeutet "WP läuft". Dann steht die Restlaufzeit im zweiten Sensor und im Attribut.
- `unknown` bedeutet "startet unter den aktuellen Bedingungen nicht" (Soll ≤ Ist), "blockiert" (Warmwasser/Abtauen) oder falscher Modus.
- Läuft die WP und Ist ≈ Soll, wird die Laufzeit auf den Deckel gesetzt und `runtime_unbounded` ist `true`.

## 5. Schritt 3: Template-Sensor Kühlen

Gleicher Aufbau, Vorzeichen gespiegelt. Als weiteren Block unter `template:` einfügen. Die Namen enthalten bewusst "Kuehlen" ohne Umlaut, damit die Entity-IDs sicher auf `..._kuehlen` enden (HA wandelt "ü" sonst in "u" um):

```yaml
  - trigger:
      - platform: time_pattern
        minutes: "/1"
      - platform: state
        entity_id:
          - sensor.nibe_degree_minutes
          - sensor.nibe_bt2_supply_temp
          - sensor.nibe_calculated_supply_temp_cooling
          - sensor.nibe_compressor_frequency
          - binary_sensor.warmwassermodus
          - binary_sensor.defrost_wp
          - sensor.nibe_operating_mode
    variables:
      mode_ok: "{{ is_state('sensor.nibe_operating_mode', 'cooling') }}"
      data_ok: >-
        {{ is_number(states('sensor.nibe_degree_minutes'))
           and is_number(states('sensor.nibe_bt2_supply_temp'))
           and is_number(states('sensor.nibe_calculated_supply_temp_cooling')) }}
      gm: "{{ states('sensor.nibe_degree_minutes') | float(0) }}"
      sup: "{{ states('sensor.nibe_bt2_supply_temp') | float(0) }}"
      req: "{{ states('sensor.nibe_calculated_supply_temp_cooling') | float(0) }}"
      running: "{{ states('sensor.nibe_compressor_frequency') | float(0) > 0 }}"
      blocked: >-
        {{ is_state('binary_sensor.warmwassermodus', 'on')
           or is_state('binary_sensor.defrost_wp', 'on') }}
      gm_start: "{{ states('input_number.wp_gm_start_kuehlen') | float(100) }}"
      gm_stop: "{{ states('input_number.wp_gm_stop_kuehlen') | float(0) }}"
      cap: "{{ states('input_number.wp_runtime_cap_min') | float(720) }}"
      need: "{{ sup - req }}"
    sensor:
      - name: "WP Minuten bis Start Kuehlen"
        unique_id: wp_minutes_to_start_cooling
        unit_of_measurement: "min"
        state_class: measurement
        state: >-
          {% if not (mode_ok and data_ok) or blocked %}
            {{ none }}
          {% elif running %}
            0
          {% elif need > 0.05 %}
            {{ [((gm_start - gm) / need) | round(0), 0] | max | int }}
          {% else %}
            {{ none }}
          {% endif %}
        attributes:
          status: >-
            {% if not (mode_ok and data_ok) %}unknown
            {% elif blocked %}blocked
            {% elif running %}running
            {% elif need > 0.05 and ((gm_start - gm) / need) <= 30 %}starting_soon
            {% else %}idle{% endif %}
          mode: cooling
          method: degree_minutes
          expected_runtime_min: >-
            {% if not (mode_ok and data_ok) or blocked or not running %}
              {{ none }}
            {% elif (req - sup) > 0.05 %}
              {{ [[((gm - gm_stop) / (req - sup)) | round(0), 0] | max, cap] | min | int }}
            {% else %}
              {{ cap | int }}
            {% endif %}
          runtime_unbounded: >-
            {{ running and (req - sup) <= 0.05 }}
          confidence: >-
            {% if not (mode_ok and data_ok) or blocked %}0
            {% elif running %}0.7
            {% else %}0.5{% endif %}
          gm_now: "{{ gm }}"
          gm_start: "{{ gm_start }}"
          gm_stop: "{{ gm_stop }}"
          supply_actual_c: "{{ sup }}"
          required_supply_c: "{{ req }}"
          updated_at: "{{ now().isoformat() }}"
      - name: "WP Laufzeit Kuehlen"
        unique_id: wp_expected_runtime_cooling
        unit_of_measurement: "min"
        state_class: measurement
        state: >-
          {% if not (mode_ok and data_ok) or blocked or not running %}
            {{ none }}
          {% elif (req - sup) > 0.05 %}
            {{ [[((gm - gm_stop) / (req - sup)) | round(0), 0] | max, cap] | min | int }}
          {% else %}
            {{ cap | int }}
          {% endif %}
```

## 5b. Variante B: reine Hysterese-WP (statt Schritt 3 bis 5)

Eine WP ohne Gradminuten startet, wenn der Ist-Vorlauf unter `Soll − h_start` fällt, und stoppt über `Soll + h_stop`. Es gibt keinen Zähler. Die Schätzung nutzt die **Steigung des Ist-Vorlaufs**:
- WP aus: `t_start = (Vorlauf − (Soll − h_start)) / (−Steigung)`
- WP läuft: `t_stop = ((Soll + h_stop) − Vorlauf) / Steigung`
- Mindestpause und Mindestlaufzeit der WP gelten als Untergrenze der Restzeit.

Das ist ungenauer als Variante A: Der Vorlauf kühlt wie eine e-Funktion aus und flacht ab, die lineare Extrapolation nennt den Start daher eher zu früh. Der Sensor setzt deshalb eine niedrige `confidence` (0,3 bis 0,4). Wenn deine WP-Dokumentation die Hysterese als "Soll + Wert" angibt, trägst du `h_start` bzw. `h_stop` einfach entsprechend ein.

**Zusätzliche Helfer:**

```yaml
input_number:
  wp_hyst_start_k:
    name: WP Hysterese Start
    min: 0
    max: 10
    step: 0.1
    mode: box
    unit_of_measurement: K
  wp_hyst_stop_k:
    name: WP Hysterese Stopp
    min: 0
    max: 10
    step: 0.1
    mode: box
    unit_of_measurement: K
  wp_min_pause_min:
    name: WP Mindestpause
    min: 0
    max: 120
    step: 1
    mode: box
    unit_of_measurement: min
  wp_min_laufzeit_min:
    name: WP Mindestlaufzeit
    min: 0
    max: 120
    step: 1
    mode: box
    unit_of_measurement: min

sensor:
  - platform: derivative
    source: sensor.nibe_bt2_supply_temp
    name: WP Vorlauf Steigung
    unit_time: min
    time_window: "00:10:00"
    round: 3
```

`sensor.wp_vorlauf_steigung` hat die Einheit K/min. Das Zeitfenster von 10 Minuten glättet Messrauschen. Für Kühlen nimmst du dieselbe Steigung.

**Template-Sensor Heizen (Variante B)**, unter `template:` einfügen (Sensornamen und Entity-IDs identisch zu Variante A, du baust pro Betriebsart nur eine der beiden Varianten; den Deckel-Helfer `wp_runtime_cap_min` aus Schritt 1 brauchst du auch hier). Der Hilfs-Binärsensor `wp_laeuft` liefert über `last_changed` die Zeit seit dem letzten Start/Stopp:

```yaml
  - binary_sensor:
      - name: "WP Laeuft"
        unique_id: wp_laeuft
        state: "{{ states('sensor.nibe_compressor_frequency') | float(0) > 0 }}"
  - trigger:
      - platform: time_pattern
        minutes: "/1"
      - platform: state
        entity_id:
          - sensor.nibe_bt2_supply_temp
          - sensor.nibe_calculated_supply_temp
          - sensor.wp_vorlauf_steigung
          - binary_sensor.wp_laeuft
          - binary_sensor.warmwassermodus
          - binary_sensor.defrost_wp
          - sensor.nibe_operating_mode
    variables:
      mode_ok: "{{ is_state('sensor.nibe_operating_mode', 'heating') }}"
      data_ok: >-
        {{ is_number(states('sensor.nibe_bt2_supply_temp'))
           and is_number(states('sensor.nibe_calculated_supply_temp'))
           and is_number(states('sensor.wp_vorlauf_steigung')) }}
      sup: "{{ states('sensor.nibe_bt2_supply_temp') | float(0) }}"
      req: "{{ states('sensor.nibe_calculated_supply_temp') | float(0) }}"
      slope: "{{ states('sensor.wp_vorlauf_steigung') | float(0) }}"
      running: "{{ is_state('binary_sensor.wp_laeuft', 'on') }}"
      blocked: >-
        {{ is_state('binary_sensor.warmwassermodus', 'on')
           or is_state('binary_sensor.defrost_wp', 'on') }}
      thr_start: "{{ req - states('input_number.wp_hyst_start_k') | float(1) }}"
      thr_stop: "{{ req + states('input_number.wp_hyst_stop_k') | float(0.5) }}"
      since_min: >-
        {{ ((now() - states.binary_sensor.wp_laeuft.last_changed).total_seconds() / 60)
           if states.binary_sensor.wp_laeuft is defined else 9999 }}
      off_wait: "{{ [states('input_number.wp_min_pause_min') | float(0) - since_min, 0] | max }}"
      on_wait: "{{ [states('input_number.wp_min_laufzeit_min') | float(0) - since_min, 0] | max }}"
      cap: "{{ states('input_number.wp_runtime_cap_min') | float(720) }}"
    sensor:
      - name: "WP Minuten bis Start Heizen"
        unique_id: wp_minutes_to_start_heating
        unit_of_measurement: "min"
        state_class: measurement
        state: >-
          {% if not (mode_ok and data_ok) or blocked %}
            {{ none }}
          {% elif running %}
            0
          {% elif sup <= thr_start %}
            {{ off_wait | round(0) | int }}
          {% elif slope < -0.005 %}
            {{ [((sup - thr_start) / (-slope)) | round(0), off_wait] | max | int }}
          {% else %}
            {{ none }}
          {% endif %}
        attributes:
          status: >-
            {% if not (mode_ok and data_ok) %}unknown
            {% elif blocked %}blocked
            {% elif running %}running
            {% elif sup <= thr_start or (slope < -0.005 and ((sup - thr_start) / (-slope)) <= 30) %}starting_soon
            {% else %}idle{% endif %}
          mode: heating
          method: hysteresis
          expected_runtime_min: >-
            {% if not (mode_ok and data_ok) or blocked or not running %}
              {{ none }}
            {% elif sup >= thr_stop %}
              {{ on_wait | round(0) | int }}
            {% elif slope > 0.005 %}
              {{ [[((thr_stop - sup) / slope) | round(0), on_wait] | max, cap] | min | int }}
            {% else %}
              {{ cap | int }}
            {% endif %}
          runtime_unbounded: "{{ running and slope <= 0.005 and sup < thr_stop }}"
          confidence: >-
            {% if not (mode_ok and data_ok) or blocked %}0
            {% elif running %}0.4
            {% else %}0.3{% endif %}
          supply_actual_c: "{{ sup }}"
          required_supply_c: "{{ req }}"
          threshold_start_c: "{{ thr_start }}"
          threshold_stop_c: "{{ thr_stop }}"
          supply_slope_k_per_min: "{{ slope }}"
          min_off_remaining_min: "{{ 0 if running else (off_wait | round(0) | int) }}"
          updated_at: "{{ now().isoformat() }}"
      - name: "WP Laufzeit Heizen"
        unique_id: wp_expected_runtime_heating
        unit_of_measurement: "min"
        state_class: measurement
        state: >-
          {% if not (mode_ok and data_ok) or blocked or not running %}
            {{ none }}
          {% elif sup >= thr_stop %}
            {{ on_wait | round(0) | int }}
          {% elif slope > 0.005 %}
            {{ [[((thr_stop - sup) / slope) | round(0), on_wait] | max, cap] | min | int }}
          {% else %}
            {{ cap | int }}
          {% endif %}
```

**Kühlen mit Hysterese** ist der gleiche Block mit diesen Änderungen:
- `climate_mode == 'cooling'`, Soll-Entität = berechneter Kühl-Vorlauf, `mode: cooling`.
- `thr_start = req + h_start` (WP startet, wenn der Vorlauf **über** dem Soll liegt), `thr_stop = req − h_stop`.
- Start: `sup >= thr_start` ergibt `off_wait`; sonst bei `slope > 0.005`: `(thr_start − sup) / slope`.
- Stopp: `sup <= thr_stop` ergibt `on_wait`; sonst bei `slope < −0.005`: `(sup − thr_stop) / (−slope)`.
- `sensor.wp_vorlauf_steigung` und `wp_laeuft` bleiben gleich.

Prüfe an einem echten Kühllauf, ob deine WP die Hysterese in dieser Richtung anwendet.

## 6. Schritt 4: Konfiguration prüfen und neu laden

1. *Entwicklerwerkzeuge → YAML → Konfiguration prüfen*, danach "Template-Entitäten neu laden" (oder HA neu starten).
2. *Entwicklerwerkzeuge → Zustände*: `sensor.wp_minuten_bis_start_heizen` suchen. Die Namen leiten sich aus den `name:`-Zeilen ab.

## 7. Schritt 5: Gegenprobe (wichtig, bevor die App den Sensor nutzt)

1. **Rechenprobe in *Entwicklerwerkzeuge → Template*:** Setze testweise `gm = 0`, `gm_start = −200`, Ist 24, Soll 26 → Ergebnis muss 100 sein. Bei laufender WP mit GM −250, Ist 27, Soll 26 → 250.
2. **Verlauf vergleichen:** Lege `sensor.wp_minuten_bis_start_heizen` und die Verdichterfrequenz in eine History-Graph-Karte. Notiere, wie viele Minuten vor einem echten Start der Sensor welchen Wert zeigte. Der Wert sollte sich linear auf 0 zubewegen. Große Sprünge zeigen, dass sich Soll oder Ist ändern (normal) oder dass deine GM-Schwellen nicht stimmen.
3. **Start- und Stoppschwelle ableiten:** GM-Wert beim echten Start und Stopp im Verlauf ablesen. Bei NIBE kommen zusätzliche Verdichterstufen bei weiteren Schwellen dazu; die Schätzung gilt nur für den ersten Start und den Stopp.
4. **Kühlen:** Vorzeichen und Schwellen der Kühl-GM an einem echten Kühllauf prüfen. Ich konnte die Konvention deiner Anlage nicht sehen. Stimmen die Vorzeichen nicht, `need`, `gm_start` und `gm_stop` im Kühlen-Block anpassen.
5. **Optional Güte messen:** Ein History-Stats- oder Statistik-Sensor über "geschätzter Start − tatsächlicher Start" zeigt, wie zuverlässig die Schätzung ist. Das Add-on macht das später selbst (Plan PR 3/4a).

## 8. Schritt 6: Anbindung an das Add-on

Das Add-on enthält den Reader (`src/hp_start_estimate.py`, Optionen `hp_start_estimate_*`, Schritt 4a-1: nur Diagnose, kein Regelverhalten). Die Nutzung in Prognose, Vorlaufsuche, Lernen, Kostenmodus und Wartezeiten folgt in den weiteren Schritten von PR 4a und ist je Verwendung einzeln schaltbar (`hp_start_estimate_use_for_*`, Standard aus). Dann in den Add-on-Optionen (Dashboard-Kategorie **"WP Load Forecast & Start Estimate"**) eintragen:
- Aktivieren: `HP_START_ESTIMATE_ENABLED = true`
- Heizen: `sensor.wp_minuten_bis_start_heizen`
- Kühlen: `sensor.wp_minuten_bis_start_kuehlen`
- Maximalalter 5 min (veraltete Werte ignoriert das Add-on).
- Restlaufzeit: Der zweite Sensor ist optional. Das Add-on liest die Restlaufzeit standardmäßig aus dem Attribut `expected_runtime_min` des Start-Sensors. Trage `sensor.wp_laufzeit_heizen` bzw. `sensor.wp_laufzeit_kuehlen` unter `hp_start_estimate_heating_runtime_entity_id` / `hp_start_estimate_cooling_runtime_entity_id` ein, hat dessen Wert Vorrang. Ist er `unknown`, `unavailable` oder ungültig, gilt das Attribut. Die Aktualität richtet sich nach dem Start-Sensor.

Bis dahin laufen die Sensoren harmlos mit und sammeln Verlauf zum Gegenprüfen.

**Unabhängigkeit im Betrieb:**
- Die Einträge sind optional. Bleibt ein Feld leer oder ist der Schalter aus, arbeitet das Add-on genau wie vorher.
- Heizen und Kühlen sind getrennt, du kannst nur einen Modus anbinden oder pro Modus eine andere Variante (A oder B) nutzen.
- Jede Verwendung (Prognose, Vorlaufsuche, Lernen, Kostenmodus, Wartezeiten) lässt sich einzeln abschalten, falls ein Bereich nicht überzeugt.
- Hysterese-Sensoren (Variante B) werden nicht automatisch gegen reale WP-Starts kalibriert. Prüfe Start- und Stoppschätzungen über mehrere echte Zyklen manuell, bevor du einzelne Verbraucher aktivierst.

## 9. Stolpersteine

- **Falscher Soll-Vorlauf:** Nimm den WP-berechneten Wert, nicht den Ausgabewert der App.
- **GM-Schwellen nicht als Sensor verfügbar:** Dann bleiben die Helfer manuell. Änderst du die Schwellen an der WP, auch die Helfer anpassen.
- **Warmwasser/Abtauen:** Während dieser Zustände ändert sich GM anders. Der Sensor meldet dann `blocked` und gibt keine Zahl aus.
- **Verdichter-Mindestlaufzeit/-pause:** Die WP kann trotz erreichter GM erst später starten. Die Schätzung ist dann zu optimistisch (Wert 0 und `idle`). Gegebenenfalls Mindestpause als Aufschlag ergänzen, wenn du das im Verlauf siehst.
- **Variante B (Hysterese):** Die Steigung ist verrauscht (Pumpentakt, Sensorsprünge). Ein Zeitfenster von 10 Minuten ist ein Kompromiss. Kürzere Fenster reagieren schneller, rauschen aber stärker. Die Schätzung gilt nur für den nächsten Start bzw. Stopp. Für Stundenprognosen ist das Tastverhältnis meist wichtiger als der einzelne Start.
- **Mehrere WP-Typen:** Nur den Block "variables" an andere Hersteller anpassen (andere Entitäten, andere Schwellenlogik). Der Vertrag zum Add-on (Zustand + Attribute) bleibt gleich.

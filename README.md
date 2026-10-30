# stimplan

Turns the manually selected stimulation electrodes of one well, each labelled with its footprint and its location
within the footprint (L1 soma, L2 axon hillock, L3 central axon, L4 terminal axon), into the stimulation table
(location, waveform, amplitude, time) for the MaxTwo.

- **Step 1 (configurations):** assigns each electrode to a routing configuration. It uses as few configurations as
  possible. Each configuration holds at most 32 electrodes, any two of its electrodes are at least `d_min` apart, and
  L1–L4 are balanced across configurations. Output: a figure.
- **Step 2 (table):** builds every row electrode × waveform × amplitude × repetition. Configurations run in random
  order, and rows are shuffled within each configuration. Each stimulus occupies one slot
  `[baseline b][stimulus d][recovery tau]`; after each re-route there is a wait `eps`.
  Output: a spreadsheet and a CSV.

Total time: `T = N (b + d + tau) + (G - 1) eps`, where `N` is the number of stimuli and `G` the number of configurations.

## Folder

```
code/
  stimplan.py          the algorithm (step 1 + step 2)
  simulate_input.py    makes a simulated electrode list (for testing only)
  batch_sim.py         runs step 1 on many simulated wells
input_example/
  electrodes_example.csv   simulated input, F = 20 footprints (80 electrodes), NOT data
output_example/
  example_configurations.png / .pdf   step 1 figure
  example_stimulation.xlsx            step 2 table + configurations + electrodes + checks
  example_stimulation.csv             step 2 table for the hardware
```

## Input

One CSV with one row per selected electrode (4F rows):

| column | required | meaning |
|---|---|---|
| `electrode` | yes | MaxWell electrode index, 0–26399 |
| `footprint` | for balancing and the figure | footprint label, e.g. f01 |
| `location` | for balancing | L1 soma, L2 axon hillock, L3 central axon, L4 terminal axon |
| `x_um`, `y_um` | optional | position in µm |

If `x_um` and `y_um` are missing, the position is computed as x = (index mod 220)·17.5 and
y = (index div 220)·17.5. **Verify this convention against one of your recordings.**

```
electrode,footprint,location
8515,f01,L1
8513,f01,L2
9380,f01,L3
7820,f01,L4
...
```

## Run

```
python3 code/stimplan.py input_example/electrodes_example.csv \
    --dmin 200 --tau 1 --b 0.35 --d 0.0004 --eps 2 --out output_example/example
```

| option | meaning | default |
|---|---|---|
| `--dmin` | minimum distance within a configuration (µm) | 200 |
| `--waveforms` | waveform names | W1,W2,W3 |
| `--amplitudes` | label=mV | A1=100,A2=200,A3=400 (placeholders) |
| `--R` | repetitions | 5 |
| `--b` | baseline before each stimulus (s) | 0.35 |
| `--d` | stimulus duration (s) | 0.0004 |
| `--tau` | recovery after each stimulus (s) | 1 |
| `--eps` | wait after each re-route (s) | 2 (placeholder) |
| `--seed` | random seed | 20261029 |
| `--out` | output prefix | well |

b, d, tau and eps must be multiples of 50 µs. Requires Python 3 with numpy, networkx, matplotlib and openpyxl.

## Output (for the example input)

Console:
```
step 1: 80 electrodes -> G = 4 configurations (lower bound 4: capacity 3, closest group 4),
        sizes {1: 20, 2: 20, 3: 20, 4: 20}, min same-config distance 216.5 um (d_min 200), L balance residual 0
        configuration 1: {'L1': 5, 'L2': 5, 'L3': 5, 'L4': 5}  (same for 2-4)
step 2: 3600 stimuli, T = 4867.44000 s = 1.3521 h
```

`example_stimulation.csv` (first rows; `time_s` is measured from the session-start marker, `time_frames` counts 50 µs frames):
```
location,waveform,amplitude_mv,time_s,time_frames,configuration
12382,W3,400.0,0.35000,7000,1
9553,W1,100.0,1.70040,34008,1
11405,W1,400.0,3.05080,61016,1
7380,W1,100.0,4.40120,88024,1
```

`example_stimulation.xlsx` sheets:
- **Parameters:** inputs, plus formula checks: closed-form T equals T from the table, the distance constraint holds,
  and every configuration has 32 or fewer electrodes.
- **Stimulation:** the table. Columns A–E are the specification; grey columns are annotations
  (configuration, footprint, L, amplitude label, rep).
- **Configurations:** each configuration's electrodes, start, end and re-route frame.
- **Electrodes:** each electrode's position, configuration, and distance to the nearest electrode in its configuration.

`example_configurations.png`: all electrodes, one colour and marker shape per configuration; below it, one panel per
configuration with discs of radius d_min/2. Discs within a configuration do not overlap.

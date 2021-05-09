# Segmentácia krvných ciev sietnice

## Inštalácia potrebných knižníc

## Spustenie segmentácie
```bash
segmentation.py -i <inputDir> -o <outputDir>
```
inputDir 
- priečinok obsahujúci snímky sietnice

outputDir 
- priečinok, kde budú uložené výsledky segmentácie
- vytvorí priečinok "results"
- vytvorí podpriečinky:
  - images - snímky celkovej segmentácie
  - thin - snímky len s tenkými cievami
  - thick - snímky len s hrubými cievami
    
- ak priečinok results existuje, nevytvára sa nový
- ak nie je zadaná cesta, priečinok sa vytvorí v inputDir

## Spustenie testu
```bash
test.py -r <resultsDir> -m <manualImagesDir>
```


"""
By: James Hassel
This file contains code needed to run the tests and output data to csv for analysis.
"""
import csv
import matchingMethods


def collectData(pairs, fileName="results.csv"):
    fImages = [pair[0] for pair in pairs]
    sImages = [pair[1] for pair in pairs]

    fieldnames = ["pairType", "fIndex", "sIndex", "templateScore", "keypointScore", "histogramScore", "hybridScore"]
    with open(fileName, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (f, s) in enumerate(pairs):
            templateScore = matchingMethods.templateMatching(f, s)
            keypointScore = matchingMethods.keyPointMatching(f, s)
            histogramScore = matchingMethods.histogramComp(f, s)
            hybridScore = matchingMethods.hybrid(f, s)
            writer.writerow({
                "pairType": "matched",
                "fIndex": idx,
                "sIndex": idx,
                "templateScore": templateScore,
                "keypointScore": keypointScore,
                "histogramScore": histogramScore,
                "hybridScore": hybridScore
            })
            print(f"Processed matched pair: fIndex={idx}, sIndex={idx} | templateScore={templateScore}, keyPointScore={keypointScore}, histogramScore={histogramScore}, hybridScore={hybridScore}")

        for f_idx, f_img in enumerate(fImages):
            # Calculate the mismatched s index using modulo to wrap around
            s_idx = (f_idx + 1) % len(sImages)
            s_img = sImages[s_idx]

            templateScore = matchingMethods.templateMatching(f_img, s_img)
            keypointScore = matchingMethods.keyPointMatching(f_img, s_img)
            histogramScore = matchingMethods.histogramComp(f_img, s_img)
            hybridScore = matchingMethods.hybrid(f, s)
            writer.writerow({
                "pairType": "mismatched",
                "fIndex": f_idx,
                "sIndex": s_idx,
                "templateScore": templateScore,
                "keypointScore": keypointScore,
                "histogramScore": histogramScore,
                "hybridScore": hybridScore
            })
            print(f"Processed mismatched pair: fIndex={f_idx}, sIndex={s_idx} | templateScore={templateScore}, keyPointScore={keypointScore}, histogramScore={histogramScore}, hybridScore={hybridScore}")



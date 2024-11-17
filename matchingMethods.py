"""
By: James Hassel
This contains the three methods I am using to determine if two images match or not.

+-----------------------+-----------------+------------------+---------------------------------------+
|Method	          	    | Accuracy	      | Speed	         |  Use Case                             |
|Template Matching		| Moderate	      | Fast	         |  When images are aligned and similar. |
|Keypoint Matching (ORB)| High	          | Moderate to Fast |	When fingerprints have distortions.  |
|Histogram Comparison	| Low to Moderate | Very Fast	     |  For quick, approximate matching.     |
+-----------------------+-----------------+------------------+---------------------------------------+
"""
import cv2
import numpy as np

def templateMatching(f, s):
    """
    Uses normalized cross-correlation for template matching between two images.
    :param f: Grayscale reference image (numpy.ndarray) -
              This is the "template" image to slide over the subject image for comparison.
              Must be a single-channel grayscale image, normalized float32 format.
    :param s: Grayscale subject image (numpy.ndarray) -
              This is the "subject" image over which the template is slid for matching.
              Must be a single-channel grayscale image, normalized float32 format.
    :return: maxVal (float) -
             A similarity score between -1 and 1, where:
             - `1` indicates a perfect match.
             - `0` indicates no correlation.
             - `-1` indicates a perfect inverse correlation.
        """
    result = cv2.matchTemplate(f, s, cv2.TM_CCOEFF_NORMED)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
    return maxVal


def keyPointMatching(f, s):
    """
    This function detects and computes keypoints and descriptors in the input images using ORB.
    It then uses a brute-force matcher with Hamming distance to match descriptors from both images.
    The result is returned as a normalized match score, which is the ratio of matches to a fixed scale (500).
    :param f: Grayscale reference image (numpy.ndarray) -
              This is the "reference" image for the matching process,
              expected to be in single-channel grayscale format uint8.
    :param s: Grayscale subject image (numpy.ndarray) -
              This is the "subject" image to compare against the reference image,
              expected to be in single-channel grayscale format uint8.
    :return: Match score (float) -
             A value between 0 and 1 representing the ratio of the number of matches found to a scale of 500.
             A higher value indicates more similarity between the images.
        """
    detector = cv2.ORB_create()

    # Ensure proper image format / Convert normalized float to uint8
    if f.dtype != np.uint8:
        f = (f * 255).astype(np.uint8)
    if s.dtype != np.uint8:
        s = (s * 255).astype(np.uint8)

    # Getting values to compare
    fkey_points, fdes = detector.detectAndCompute(f, None)
    skey_points, sdes = detector.detectAndCompute(s, None)

    # Doing the comparison
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(fdes, sdes)
    return len(matches) / 500


def histogramComp(f, s):
    """
    This function computes the histograms of the input images and compares them using
    OpenCV's `cv2.compareHist` with the correlation method. The result indicates the similarity
    between the two histograms, where 1 indicates perfect similarity.
    :param f: Grayscale reference image (numpy.ndarray) -
              This is the "reference" image for the matching process,
              expected to be in single-channel grayscale format uint8.
    :param s: Grayscale subject image (numpy.ndarray) -
              This is the "subject" image to compare against the reference image,
              expected to be in single-channel grayscale format uint8.
    :return: score (float) -
             A similarity score between -1 and 1, where:
             - `1`: Identical histograms.
             - `0`: No correlation.
             - `-1`: Perfect inverse correlation.
        """
    # Ensure the images are in uint8 format
    if f.dtype != np.uint8:
        f = (f * 255).astype(np.uint8)
    if s.dtype != np.uint8:
        s = (s * 255).astype(np.uint8)

    # Compute histograms + normalize
    hist_f = cv2.calcHist([f], [0], None, [256], [0, 256])
    hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
    hist_f = cv2.normalize(hist_f, hist_f).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()

    # Compare histograms and return score
    return cv2.compareHist(hist_f, hist_s, cv2.HISTCMP_CORREL)


def hybrid(f, s):
    """
    Combines the results of templateMatching, keyPointMatching, and histogramComp
    to calculate a confidence score between 0 and 1 for how likely two images match.
    :param f: Grayscale reference image (numpy.ndarray).
    :param s: Grayscale subject image (numpy.ndarray).
    :return: Confidence score (float) -
             A value between 0 and 1, where:
             - `1` indicates a very confident match.
             - `0` indicates no confidence in a match.
        """
    # Thresholds for each method
    template_threshold = 0.7
    keypoint_threshold = 0.1
    histogram_threshold = 0.7

    template_score = templateMatching(f, s)
    keypoint_score = keyPointMatching(f, s)
    histogram_score = histogramComp(f, s)

    # Normalize
    template_conf = min(1, max(0, (template_score - template_threshold) / (1 - template_threshold)))
    keypoint_conf = min(1, max(0, (keypoint_score - keypoint_threshold) / (1 - keypoint_threshold)))
    histogram_conf = min(1, max(0, (histogram_score - histogram_threshold) / (1 - histogram_threshold)))

    return (template_conf + keypoint_conf + histogram_conf) / 3

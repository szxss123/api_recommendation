# Strict New-Mashup Cold-Start Leakage and Near-Duplicate Audit

This audit separates three different phenomena:

1. exact/near duplicate Mashups across train and test, which can threaten the validity of a strict split;
2. direct mentions of a ground-truth API name in test Mashup text, which are legal side information only if the deployment setting exposes such text;
3. API overlap between a test Mashup and its nearest training Mashups, which is the intended transfer signal of the inductive collaborative baselines and is not by itself leakage.

## Summary

| Item | Value |
| --- | --- |
| Test Mashups | 1645 |
| Exact train-text duplicates | 4 |
| Exact train-name duplicates | 95 |
| Mashups mentioning ≥1 ground-truth API name | 705 (42.86%) |
| Top-1 neighbor sharing ≥1 positive API | 1114 (67.72%) |
| Mean Top-1 shared-positive recall | 0.5299 |
| Mean Top-K union positive recall | 0.7904 |
| Mean / median / max Top-1 cosine | 0.7926 / 0.7796 / 0.9979 |

## Near-duplicate thresholds

- Top-1 cosine ≥ 0.90: 148
- Top-1 cosine ≥ 0.95: 118
- Top-1 cosine ≥ 0.98: 78
- Top-1 cosine ≥ 0.99: 42

## Highest-risk cases

| mashup_id | mashup_identifier | exact_train_text_duplicate | exact_train_name_duplicate | top1_train_mashup_identifier | top1_cosine_similarity | top1_shared_positive_recall | direct_api_name_mention_recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6028 | qmpeople-city-description | 1 | 1 | qmpeople-city-description- | 0.9969 | 1.0000 | 1.0000 |
| 327 | world-of-facebook-friends. | 1 | 1 | world-of-facebook-friends | 0.9972 | 1.0000 | 0.2500 |
| 480 | omg-rainbows! | 1 | 1 | omg-rainbows | 0.9944 | 1.0000 | 0.0000 |
| 4410 | yahoo!-apis-example-with-code | 1 | 1 | yahoo-apis-example-with-code | 0.9886 | 1.0000 | 0.0000 |
| 7906 | retrievr-search-by-sketch | 0 | 1 | retrievr---search-by-sketch | 0.9962 | 1.0000 | 1.0000 |
| 7762 | google-maps-fast-food | 0 | 1 | google-maps-+-fast-food | 0.9955 | 1.0000 | 1.0000 |
| 8132 | possibly-related-classroom-projects-wordpress | 0 | 1 | possibly-related-classroom-projects-(wordpress) | 0.9942 | 1.0000 | 1.0000 |
| 7448 | flickorama-3d-flickr-photoset-viewer | 0 | 1 | flickorama:-3d-flickr-photoset-viewer | 0.9935 | 1.0000 | 1.0000 |
| 3717 | tumblr-+-twilio-voice-posts | 0 | 1 | tumblr-twilio-voice-posts | 0.9931 | 1.0000 | 1.0000 |
| 5489 | new-york-art-beat---bubble-machine | 0 | 1 | new-york-art-beat-bubble-machine | 0.9926 | 1.0000 | 1.0000 |
| 6351 | googawho-side-by-side-search | 0 | 1 | googawho?-side-by-side-search | 0.9917 | 1.0000 | 1.0000 |
| 5299 | cyborg-karaoke-party- | 0 | 1 | cyborg-karaoke-party | 0.9909 | 1.0000 | 1.0000 |
| 7835 | social-page-authority-checker | 0 | 1 | -social-page-authority-checker | 0.9900 | 0.7500 | 1.0000 |
| 4315 | droidin---linkedin-on-android | 0 | 1 | droidin-linkedin-on-android | 0.9855 | 1.0000 | 1.0000 |
| 4418 | bp-gulf-oil-spill---view-in-google-earth | 0 | 1 | bp-gulf-oil-spill-view-in-google-earth | 0.9854 | 1.0000 | 1.0000 |
| 7015 | turkish-real-estate-search-powered-by-google-maps | 0 | 1 | turkish-real-estate-search---powered-by-google-maps | 0.9828 | 1.0000 | 1.0000 |
| 5689 | molu-abbreviations-bot- | 0 | 1 | molu-abbreviations-bot | 0.9818 | 1.0000 | 1.0000 |
| 5232 | raspberry-pi-and-plivo-â€”-call-mom-button | 0 | 1 | raspberry-pi-and-plivo-call-mom-button | 0.9805 | 1.0000 | 1.0000 |
| 4828 | tarpipe-for-evernote- | 0 | 1 | tarpipe-for-evernote | 0.9803 | 1.0000 | 1.0000 |
| 4684 | etsyhacks:-where-am-i? | 0 | 1 | etsyhacks-where-am-i | 0.9782 | 1.0000 | 1.0000 |

## Interpretation rules

- Any exact text/name duplicate should be manually inspected.
- A very high cosine score alone is not proof of leakage; inspect the text.
- API-name mentions must be disclosed in the paper if names/descriptions are available to the recommender at inference time.
- Shared APIs among semantic neighbors explain why inductive collaborative transfer can outperform direct text similarity.

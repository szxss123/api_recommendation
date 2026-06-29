# Strict Cold-Start Case Study Report with Names

The recommendation lists are consensus Top-10 rankings aggregated over seeds 0/1/2 by reciprocal-rank voting.

## Graph+BGE succeeds while Graph-only fails

**Mashup:** Anaphonr (`anaphonr`, internal ID `3781`)

**Description:** fun mashup help find anaphones anaphone word phrase name formed rearranging sound another galaxy lucky gas plus power twilio text anaphone inquiry 785 378 5626

**Selection reason:** Graph+BGE hits relevant APIs in at least two seeds, while Graph-only misses in all seeds.

**Mean metrics:** Graph+BGE R@10=1.0000, NDCG@10=1.0000; Graph-only R@10=0.0000, NDCG@10=0.0000; BGE-only R@10=0.0000, NDCG@10=0.0000.

### Ground truth

| API ID | API name | API slug | Group | Train frequency |
| --- | --- | --- | --- | --- |
| 13 | Twilio SMS | twilio-sms | Head | 141 |

### Popularity

| Rank | API ID | API name | Group | Train freq. | Hit | Seeds |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 624 | Google Maps | Head | 1898 |  | 3 |
| 2 | 1079 | Twitter | Head | 629 |  | 3 |
| 3 | 1069 | Youtube | Head | 513 |  | 3 |
| 4 | 90 | Flickr | Head | 458 |  | 3 |
| 5 | 867 | Facebook | Head | 354 |  | 3 |
| 6 | 96 | Amazon Product Advertising | Head | 322 |  | 3 |
| 7 | 712 | Twilio | Head | 270 |  | 3 |
| 8 | 1401 | Last Fm | Head | 178 |  | 3 |
| 9 | 634 | Ebay | Head | 164 |  | 3 |
| 10 | 13 | Twilio SMS | Head | 141 | ✓ | 3 |

### Graph-only

| Rank | API ID | API name | Group | Train freq. | Hit | Seeds |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 624 | Google Maps | Head | 1898 |  | 3 |
| 2 | 1079 | Twitter | Head | 629 |  | 3 |
| 3 | 1069 | Youtube | Head | 513 |  | 3 |
| 4 | 90 | Flickr | Head | 458 |  | 3 |
| 5 | 867 | Facebook | Head | 354 |  | 3 |
| 6 | 96 | Amazon Product Advertising | Head | 322 |  | 3 |
| 7 | 1401 | Last Fm | Head | 178 |  | 3 |
| 8 | 634 | Ebay | Head | 164 |  | 3 |
| 9 | 712 | Twilio | Head | 270 |  | 3 |
| 10 | 204 | Google Search | Head | 138 |  | 1 |

### BGE-only

| Rank | API ID | API name | Group | Train freq. | Hit | Seeds |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 133 | Spokenbuzz | Tail | 1 |  | 3 |
| 2 | 723 | Plum Fuse | Middle | 3 |  | 3 |
| 3 | 639 | Backtweets | Middle | 4 |  | 3 |
| 4 | 362 | Rhymebrain | Tail | 1 |  | 3 |
| 5 | 931 | Nabaztag | Tail | 1 |  | 3 |
| 6 | 202 | Snappyfingers | Tail | 1 |  | 3 |
| 7 | 406 | Maluuba Natural Language | Tail | 1 |  | 3 |
| 8 | 1507 | Shoutcast Radio | Tail | 1 |  | 3 |
| 9 | 1561 | Touchtunes Jukebox | Tail | 1 |  | 3 |
| 10 | 717 | Tringme | Tail | 1 |  | 3 |

### Graph+BGE

| Rank | API ID | API name | Group | Train freq. | Hit | Seeds |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 13 | Twilio SMS | Head | 141 | ✓ | 3 |
| 2 | 712 | Twilio | Head | 270 |  | 3 |
| 3 | 314 | 411sync | Head | 85 |  | 3 |
| 4 | 1079 | Twitter | Head | 629 |  | 3 |
| 5 | 1401 | Last Fm | Head | 178 |  | 3 |
| 6 | 284 | Lyricwiki | Head | 19 |  | 3 |
| 7 | 1069 | Youtube | Head | 513 |  | 2 |
| 8 | 262 | Lyricsfly | Head | 18 |  | 2 |
| 9 | 90 | Flickr | Head | 458 |  | 2 |
| 10 | 1171 | Trynt | Head | 14 |  | 2 |

## Graph+BGE retrieves a Middle positive API

**Mashup:** 30 Boxes Evoketv (`30-boxes-evoketv`, internal ID `8128`)

**Description:** mashup technique adding evoketv listing automoatically 30 box calendar

**Selection reason:** The consensus Graph+BGE Top-10 contains a relevant Middle API.

**Mean metrics:** Graph+BGE R@10=1.0000, NDCG@10=0.8333; Graph-only R@10=0.0000, NDCG@10=0.0000; BGE-only R@10=1.0000, NDCG@10=1.0000.

### Ground truth

| API ID | API name | API slug | Group | Train frequency |
| --- | --- | --- | --- | --- |
| 428 | 30 Boxes | 30-boxes | Middle | 3 |

### Popularity

| Rank | API ID | API name | Group | Train freq. | Hit | Seeds |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 624 | Google Maps | Head | 1898 |  | 3 |
| 2 | 1079 | Twitter | Head | 629 |  | 3 |
| 3 | 1069 | Youtube | Head | 513 |  | 3 |
| 4 | 90 | Flickr | Head | 458 |  | 3 |
| 5 | 867 | Facebook | Head | 354 |  | 3 |
| 6 | 96 | Amazon Product Advertising | Head | 322 |  | 3 |
| 7 | 712 | Twilio | Head | 270 |  | 3 |
| 8 | 1401 | Last Fm | Head | 178 |  | 3 |
| 9 | 634 | Ebay | Head | 164 |  | 3 |
| 10 | 13 | Twilio SMS | Head | 141 |  | 3 |

### Graph-only

| Rank | API ID | API name | Group | Train freq. | Hit | Seeds |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 624 | Google Maps | Head | 1898 |  | 3 |
| 2 | 1079 | Twitter | Head | 629 |  | 3 |
| 3 | 1069 | Youtube | Head | 513 |  | 3 |
| 4 | 90 | Flickr | Head | 458 |  | 3 |
| 5 | 867 | Facebook | Head | 354 |  | 3 |
| 6 | 96 | Amazon Product Advertising | Head | 322 |  | 3 |
| 7 | 1401 | Last Fm | Head | 178 |  | 3 |
| 8 | 634 | Ebay | Head | 164 |  | 3 |
| 9 | 712 | Twilio | Head | 270 |  | 3 |
| 10 | 204 | Google Search | Head | 138 |  | 1 |

### BGE-only

| Rank | API ID | API name | Group | Train freq. | Hit | Seeds |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 428 | 30 Boxes | Middle | 3 | ✓ | 3 |
| 2 | 69 | Evoca | Tail | 1 |  | 3 |
| 3 | 1630 | 5min | Tail | 1 |  | 3 |
| 4 | 1445 | Backpack | Middle | 3 |  | 3 |
| 5 | 318 | Eventful | Head | 32 |  | 3 |
| 6 | 1427 | Producteev | Tail | 2 |  | 3 |
| 7 | 524 | Wishpot Shopping | Tail | 1 |  | 3 |
| 8 | 1123 | Doodle | Tail | 1 |  | 3 |
| 9 | 1288 | Cnet | Head | 13 |  | 3 |
| 10 | 1024 | Vodpod | Middle | 2 |  | 3 |

### Graph+BGE

| Rank | API ID | API name | Group | Train freq. | Hit | Seeds |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 428 | 30 Boxes | Middle | 3 | ✓ | 3 |
| 2 | 1069 | Youtube | Head | 513 |  | 3 |
| 3 | 624 | Google Maps | Head | 1898 |  | 3 |
| 4 | 271 | Google Calendar | Head | 30 |  | 3 |
| 5 | 318 | Eventful | Head | 32 |  | 2 |
| 6 | 413 | Google Homepage | Head | 77 |  | 3 |
| 7 | 1079 | Twitter | Head | 629 |  | 2 |
| 8 | 1401 | Last Fm | Head | 178 |  | 3 |
| 9 | 735 | Box | Head | 62 |  | 2 |
| 10 | 1288 | Cnet | Head | 13 |  | 2 |

## Graph+BGE retrieves a Tail positive API

**Mashup:** Pdxster (`pdxster`, internal ID `5307`)

**Description:** view comment portland oregon city council agenda

**Selection reason:** The consensus Graph+BGE Top-10 contains a relevant Tail API.

**Mean metrics:** Graph+BGE R@10=0.6667, NDCG@10=0.3102; Graph-only R@10=0.0000, NDCG@10=0.0000; BGE-only R@10=1.0000, NDCG@10=1.0000.

### Ground truth

| API ID | API name | API slug | Group | Train frequency |
| --- | --- | --- | --- | --- |
| 249 | Pdxcouncilconnect | pdxcouncilconnect | Tail | 1 |

### Popularity

| Rank | API ID | API name | Group | Train freq. | Hit | Seeds |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 624 | Google Maps | Head | 1898 |  | 3 |
| 2 | 1079 | Twitter | Head | 629 |  | 3 |
| 3 | 1069 | Youtube | Head | 513 |  | 3 |
| 4 | 90 | Flickr | Head | 458 |  | 3 |
| 5 | 867 | Facebook | Head | 354 |  | 3 |
| 6 | 96 | Amazon Product Advertising | Head | 322 |  | 3 |
| 7 | 712 | Twilio | Head | 270 |  | 3 |
| 8 | 1401 | Last Fm | Head | 178 |  | 3 |
| 9 | 634 | Ebay | Head | 164 |  | 3 |
| 10 | 13 | Twilio SMS | Head | 141 |  | 3 |

### Graph-only

| Rank | API ID | API name | Group | Train freq. | Hit | Seeds |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 624 | Google Maps | Head | 1898 |  | 3 |
| 2 | 1079 | Twitter | Head | 629 |  | 3 |
| 3 | 1069 | Youtube | Head | 513 |  | 3 |
| 4 | 90 | Flickr | Head | 458 |  | 3 |
| 5 | 867 | Facebook | Head | 354 |  | 3 |
| 6 | 96 | Amazon Product Advertising | Head | 322 |  | 3 |
| 7 | 1401 | Last Fm | Head | 178 |  | 3 |
| 8 | 634 | Ebay | Head | 164 |  | 3 |
| 9 | 712 | Twilio | Head | 270 |  | 3 |
| 10 | 204 | Google Search | Head | 138 |  | 1 |

### BGE-only

| Rank | API ID | API name | Group | Train freq. | Hit | Seeds |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 249 | Pdxcouncilconnect | Tail | 1 | ✓ | 3 |
| 2 | 608 | Trimet | Tail | 1 |  | 3 |
| 3 | 415 | Citysourced | Unseen | 0 |  | 3 |
| 4 | 272 | Govtrack US | Head | 8 |  | 3 |
| 5 | 744 | Google Civic Information | Tail | 1 |  | 3 |
| 6 | 794 | Transparency Data | Tail | 1 |  | 3 |
| 7 | 1135 | Theyworkforyou | Unseen | 0 |  | 3 |
| 8 | 192 | Maplight | Middle | 2 |  | 3 |
| 9 | 957 | Newscloud | Middle | 2 |  | 3 |
| 10 | 518 | Metro Realtime | Tail | 1 |  | 3 |

### Graph+BGE

| Rank | API ID | API name | Group | Train freq. | Hit | Seeds |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1056 | Technorati | Head | 45 |  | 2 |
| 2 | 481 | Sunlight Labs Congress | Head | 18 |  | 3 |
| 3 | 624 | Google Maps | Head | 1898 |  | 2 |
| 4 | 272 | Govtrack US | Head | 8 |  | 3 |
| 5 | 249 | Pdxcouncilconnect | Tail | 1 | ✓ | 2 |
| 6 | 1079 | Twitter | Head | 629 |  | 2 |
| 7 | 90 | Flickr | Head | 458 |  | 2 |
| 8 | 867 | Facebook | Head | 354 |  | 2 |
| 9 | 770 | Opensecrets | Head | 7 |  | 2 |
| 10 | 122 | Del Icio US | Head | 105 |  | 2 |

## BGE-only provides useful semantic evidence

**Mashup:** Docusign For Outlook (`docusign-for-outlook`, internal ID `4059`)

**Description:** docusign outlook esignature application fully integrated outlook application user able electronically sign document sent email andor assign people sign document within contact

**Selection reason:** BGE-only finds relevant semantic evidence that Graph-only misses; fusion retains useful evidence.

**Mean metrics:** Graph+BGE R@10=1.0000, NDCG@10=1.0000; Graph-only R@10=0.0000, NDCG@10=0.0000; BGE-only R@10=1.0000, NDCG@10=1.0000.

### Ground truth

| API ID | API name | API slug | Group | Train frequency |
| --- | --- | --- | --- | --- |
| 547 | Docusign Enterprise | docusign-enterprise | Head | 68 |

### Popularity

| Rank | API ID | API name | Group | Train freq. | Hit | Seeds |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 624 | Google Maps | Head | 1898 |  | 3 |
| 2 | 1079 | Twitter | Head | 629 |  | 3 |
| 3 | 1069 | Youtube | Head | 513 |  | 3 |
| 4 | 90 | Flickr | Head | 458 |  | 3 |
| 5 | 867 | Facebook | Head | 354 |  | 3 |
| 6 | 96 | Amazon Product Advertising | Head | 322 |  | 3 |
| 7 | 712 | Twilio | Head | 270 |  | 3 |
| 8 | 1401 | Last Fm | Head | 178 |  | 3 |
| 9 | 634 | Ebay | Head | 164 |  | 3 |
| 10 | 13 | Twilio SMS | Head | 141 |  | 3 |

### Graph-only

| Rank | API ID | API name | Group | Train freq. | Hit | Seeds |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 624 | Google Maps | Head | 1898 |  | 3 |
| 2 | 1079 | Twitter | Head | 629 |  | 3 |
| 3 | 1069 | Youtube | Head | 513 |  | 3 |
| 4 | 90 | Flickr | Head | 458 |  | 3 |
| 5 | 867 | Facebook | Head | 354 |  | 3 |
| 6 | 96 | Amazon Product Advertising | Head | 322 |  | 3 |
| 7 | 1401 | Last Fm | Head | 178 |  | 3 |
| 8 | 634 | Ebay | Head | 164 |  | 3 |
| 9 | 712 | Twilio | Head | 270 |  | 3 |
| 10 | 204 | Google Search | Head | 138 |  | 1 |

### BGE-only

| Rank | API ID | API name | Group | Train freq. | Hit | Seeds |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 547 | Docusign Enterprise | Head | 68 | ✓ | 3 |
| 2 | 1531 | Echosign | Middle | 3 |  | 3 |
| 3 | 909 | Enthusem | Tail | 1 |  | 3 |
| 4 | 1096 | Endicia Label Server | Middle | 3 |  | 3 |
| 5 | 1364 | Eduroam | Tail | 1 |  | 3 |
| 6 | 113 | Mandrill | Middle | 2 |  | 3 |
| 7 | 557 | Constantcontact | Unseen | 0 |  | 3 |
| 8 | 564 | Issuu Search | Middle | 3 |  | 3 |
| 9 | 542 | Interfax | Middle | 2 |  | 3 |
| 10 | 690 | Mailjet | Tail | 1 |  | 3 |

### Graph+BGE

| Rank | API ID | API name | Group | Train freq. | Hit | Seeds |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 547 | Docusign Enterprise | Head | 68 | ✓ | 3 |
| 2 | 1069 | Youtube | Head | 513 |  | 3 |
| 3 | 1531 | Echosign | Middle | 3 |  | 3 |
| 4 | 314 | 411sync | Head | 85 |  | 2 |
| 5 | 96 | Amazon Product Advertising | Head | 322 |  | 3 |
| 6 | 624 | Google Maps | Head | 1898 |  | 2 |
| 7 | 1401 | Last Fm | Head | 178 |  | 3 |
| 8 | 867 | Facebook | Head | 354 |  | 2 |
| 9 | 122 | Del Icio US | Head | 105 |  | 3 |
| 10 | 1079 | Twitter | Head | 629 |  | 2 |

## Hard multi-API failure with unseen services

**Mashup:** Kissapi (`kissapi`, internal ID `4154`)

**Description:** site let choose facebook picture automatically create collage post onto facebook order picture item like mug pillow necklace etc

**Selection reason:** The Mashup has six positive APIs, including two Unseen services, and all methods miss the entire positive set at Top-10.

**Mean metrics:** Graph+BGE R@10=0.0000, NDCG@10=0.0000; Graph-only R@10=0.0000, NDCG@10=0.0000; BGE-only R@10=0.0000, NDCG@10=0.0000.

### Ground truth

| API ID | API name | API slug | Group | Train frequency |
| --- | --- | --- | --- | --- |
| 95 | Hp Cloud Object Storage | hp-cloud-object-storage | Unseen | 0 |
| 570 | Facebook Graph | facebook-graph | Head | 38 |
| 643 | Hp Labs Multimedia Analytic Platform | hp-labs-multimedia-analytic-platform | Unseen | 0 |
| 976 | Amazon S3 | amazon-s3 | Head | 77 |
| 1381 | Zazzle | zazzle | Head | 14 |
| 1588 | Amazon EC2 | amazon-ec2 | Head | 59 |

### Popularity

| Rank | API ID | API name | Group | Train freq. | Hit | Seeds |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 624 | Google Maps | Head | 1898 |  | 3 |
| 2 | 1079 | Twitter | Head | 629 |  | 3 |
| 3 | 1069 | Youtube | Head | 513 |  | 3 |
| 4 | 90 | Flickr | Head | 458 |  | 3 |
| 5 | 867 | Facebook | Head | 354 |  | 3 |
| 6 | 96 | Amazon Product Advertising | Head | 322 |  | 3 |
| 7 | 712 | Twilio | Head | 270 |  | 3 |
| 8 | 1401 | Last Fm | Head | 178 |  | 3 |
| 9 | 634 | Ebay | Head | 164 |  | 3 |
| 10 | 13 | Twilio SMS | Head | 141 |  | 3 |

### Graph-only

| Rank | API ID | API name | Group | Train freq. | Hit | Seeds |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 624 | Google Maps | Head | 1898 |  | 3 |
| 2 | 1079 | Twitter | Head | 629 |  | 3 |
| 3 | 1069 | Youtube | Head | 513 |  | 3 |
| 4 | 90 | Flickr | Head | 458 |  | 3 |
| 5 | 867 | Facebook | Head | 354 |  | 3 |
| 6 | 96 | Amazon Product Advertising | Head | 322 |  | 3 |
| 7 | 1401 | Last Fm | Head | 178 |  | 3 |
| 8 | 634 | Ebay | Head | 164 |  | 3 |
| 9 | 712 | Twilio | Head | 270 |  | 3 |
| 10 | 413 | Google Homepage | Head | 77 |  | 1 |

### BGE-only

| Rank | API ID | API name | Group | Train freq. | Hit | Seeds |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1439 | Imgur | Tail | 2 |  | 3 |
| 2 | 1010 | Cheezburger | Middle | 2 |  | 3 |
| 3 | 511 | Wookmark | Tail | 1 |  | 3 |
| 4 | 1495 | Dipity | Middle | 4 |  | 3 |
| 5 | 1126 | Bebo | Head | 6 |  | 3 |
| 6 | 228 | Eyeem | Tail | 1 |  | 3 |
| 7 | 456 | Sharethis | Unseen | 0 |  | 3 |
| 8 | 1376 | Peekyou Social Analytics | Middle | 2 |  | 3 |
| 9 | 1112 | Pinboard | Middle | 2 |  | 3 |
| 10 | 391 | Webshots | Middle | 2 |  | 3 |

### Graph+BGE

| Rank | API ID | API name | Group | Train freq. | Hit | Seeds |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 867 | Facebook | Head | 354 |  | 3 |
| 2 | 90 | Flickr | Head | 458 |  | 3 |
| 3 | 1079 | Twitter | Head | 629 |  | 3 |
| 4 | 1069 | Youtube | Head | 513 |  | 3 |
| 5 | 122 | Del Icio US | Head | 105 |  | 3 |
| 6 | 413 | Google Homepage | Head | 77 |  | 3 |
| 7 | 1264 | Friendfeed | Head | 24 |  | 3 |
| 8 | 624 | Google Maps | Head | 1898 |  | 2 |
| 9 | 1401 | Last Fm | Head | 178 |  | 2 |
| 10 | 1056 | Technorati | Head | 45 |  | 1 |

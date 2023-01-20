"""
This is the data obtained from other works related in this sense is used to generate
data than represent the data possible non seen to train with it.

1 -- GoogleDrive
2 -- OneDrive
3 -- DropBox
4 -- Box
5 -- Egnyte
6 -- Sharefile
7 -- Salesforce
8 -- Alibaba cloud
9 -- Amazon Cloud Drive
10 - Apple iCloud
11 - Azure Storage

"""

CLOUD_NAMES = {
    0: "GoogleDrive",
    1: "OneDrive",
    2: "DropBox",
    3: "Box",
    4: "Egnyte",
    5: "ShareFile",
    6: "SalesForce",
    7: "Alibaba cloud",
    8: " Amazon Cloud Drive",
    9: "Apple iCloud",
    10: "Azure Storage",
}

FAILURE_PROBABILITY = {
    0: (0.00072679, 0.00145361),
    1: (0.00066020, 0.00132043),
    2: (0.00097032, 0.00194065),
    3: (0.00179699, 0.00359402),
    4: (0.00073249, 0.00146503),
    5: (0.00014269, 0.00028542),
    6: (0.00061739, 0.00123480),
    7: (0.00079897, 0.00138793),
    8: (0.00018227, 0.00098241),
    9: (0.00015664, 0.00097632),
    10: (0.00039977, 0.00087241),
}

DOWLOAD_SPEED = {
    0: (2.15, 3.26),
    1: (1.21, 2.41),
    2: (3.07, 3.32),
    3: (2.01, 3.20),
    4: (2.17, 2.36),
    5: (0.72, 0.76),
    6: (0.68, 0.72),
    7: (2.54, 3.18),
    8: (2.49, 3.09),
    9: (2.01, 2.98),
    10: (2.30, 3.12),
}

UPLOAD_SPEED = {
    0: (1.79, 3.24),
    1: (0.91, 1.70),
    2: (2.59, 3.05),
    3: (1.91, 3.27),
    4: (1.24, 1.93),
    5: (0.11, 0.65),
    6: (0.52, 0.73),
    7: (2.32, 3.14),
    8: (0.70, 1.86),
    9: (2.05, 3.45),
    10: (1.31, 3.17),
}


REDUNDANCY = {
    "22": 2,
    "23": 3,
    "33": 2,
    "24": 4,
    "34": 2.67,
    "44": 2,
    "25": 5,
    "35": 3.3,
    "45": 2.5,
    "55": 2,
    "26": 6,
    "36": 4,
    "46": 3,
    "56": 2.4,
    "66": 2,
    "27": 7,
    "37": 4.67,
    "47": 3.50,
    "57": 2.80,
    "67": 2.33,
    "77": 2,
    "28": 8,
    "38": 5.33,
    "48": 4.00,
    "58": 3.20,
    "68": 2.67,
    "78": 2.29,
    "88": 2,
    "29": 9,
    "39": 6,
    "49": 4.5,
    "59": 3.60,
    "69": 3.00,
    "79": 2.57,
    "89": 2.25,
    "99": 2,
    "210": 10,
    "310": 6.67,
    "410": 5,
    "510": 4,
    "610": 3.33,
    "710": 2.86,
    "810": 2.50,
    "910": 2.22,
    "1010": 2.00,
    "211": 11,
    "311": 7.33,
    "411": 5.50,
    "511": 4.40,
    "611": 3.67,
    "711": 3.14,
    "811": 2.75,
    "911": 2.44,
    "1011": 2.20,
    "1111": 2,
}

TIME = {
    "22": 255,
    "23": 275,
    "33": 267,
    "24": 354,
    "34": 296,
    "44": 269,
    "25": 472.91,
    "35": 354.24,
    "45": 326.80,
    "55": 292.93,
    "26": 567.10,
    "36": 433.62,
    "46": 368.28,
    "56": 333.78,
    "66": 293.20,
    "27": 636.15,
    "37": 485.77,
    "47": 412.74,
    "57": 374.42,
    "67": 337.15,
    "77": 306.36,
    "28": 756.47,
    "38": 552.41,
    "48": 455.53,
    "58": 394.46,
    "68": 369.37,
    "78": 323.95,
    "88": 303.01,
    "29": 806.42,
    "39": 605.95,
    "49": 493.77,
    "59": 441.51,
    "69": 409.11,
    "79": 350.38,
    "89": 328.97,
    "99": 301.11,
    "210": 838.49,
    "310": 614.82,
    "410": 520.69,
    "510": 436.35,
    "610": 396.58,
    "710": 358.53,
    "810": 345.72,
    "910": 314.36,
    "1010": 295.72,
    "211": 914.58,
    "311": 646.07,
    "411": 529.79,
    "511": 474.61,
    "611": 412.24,
    "711": 386.46,
    "811": 356.12,
    "911": 329.70,
    "1011": 301.23,
    "1111": 292.26,
}

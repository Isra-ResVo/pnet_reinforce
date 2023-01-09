

r"""
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
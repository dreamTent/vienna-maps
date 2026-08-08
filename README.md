# Vienna Maps
A Birds eye view of vienna in high resolution.

This project uses aerial photography created and distributed by the city of vienna and available over their open data platform under the CC BY 4.0 license.

## Disclaimer

This Project has been vibe coded and is highly inefficient.  
__This Project is work in progress!!__

## How to use

Visit: https://dreamtent.dev/projects/vienna-map/


## Setup (only for infinite scroll)
The setup applies only if you want to build uppon this project.  
And or want to host the cdn files yourself.  
The Ortho images for infinite-scoll are currently intended to be hosted together with the application, but are not in the repo.  
They would have to be downloaded seperately and put into /orthofotos/2023  
Download here:  https://cdn.map.dreamtent.dev/data/orthofoto/vienna/2023/s0.zip  
Or you can edit the source to load them from the cdn directly. See source code for instructions.  


# What else is in this repo?
- Scrips for downloading the original images
- Scripts for downloading the original images covering a specific area and creating a colmap
- Viewing the camera positions of oblique images for 2020 and 2023



## Datasources Oblique Images
This project uses a mirrored dataset for oblique images under:  
https://cdn.map.dreamtent.dev/data/oblique/vienna/2023/  
and:  
https://cdn.map.dreamtent.dev/data/oblique/vienna/2020/  

The original images are available under:  
2023:  
https://www.data.gv.at/datasets/7fa23581-0df3-499e-8593-e80201e4825c?locale=de
https://www.wien.gv.at/stadtplan3d/datasource-data/Oblique/91bf9860-34d7-4f9a-b841-32be753b09e5  
2020:  
https://www.data.gv.at/datasets/2aada6af-9aaa-42ae-98f2-8d974b278280?locale=de  
https://www.wien.gv.at/stadtplan3d/datasource-data/Oblique/Oblique_Wien_2020  
But they are very slow.


## Datasources Orthofoto Images
Available under the correct format and size for the application under:  
https://cdn.map.dreamtent.dev/data/orthofoto/vienna/2023/  

Original:  
https://www.data.gv.at/datasets/b2d2433d-2a39-46e4-9f9d-57c14e8b2408
https://www.wien.gv.at/ma41datenviewer/public/start.aspx
# soccer_betting
Implementing a betting strategy using Fifa data

The soccer_project file is used to gather data from various SQLite DB. It then organises it in such a way that it is easier for the following files to run a model on

soccer_model and soccer_model_penalized are two similar models, but use different optimization objective. Turns out the simplest one (the first one) works better, but I wanted to keep the other one here because it was more "original"

soccer_betting_strategy basically tries to answer the following question : should I bet on the outcome I think is most likely, or on the one I think the bookmaker has given overestimated odds to?

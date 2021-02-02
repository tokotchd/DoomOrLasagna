# DoomOrLasagna

By the power of a $1 domain registration (GoDaddy.com) and AWS's Free tier...<br>
A Running instance of this website/portfolio be found at http://DoomOrLasagna.site <br>

#Things I would have done differently
- So. Much. More. Security.  
  - This application has close to zero good security practices, due to the self-imposed time constraints.
  - SSL, hosting by proxy, removing page references to the EC2 IP, supporting  ecrypted uploads, hosting deployed package instead of source, etc.
- Bigger, Better Model?
  - The model chosen and trained was due to development constraints (Needed to be fast to train, no GPU resources, had to be small enough to run on EC2 free tier, etc.)
  - Chances are a bigger model could have been developed and ran on the EC2 instance.  At worst, more time specifically working on the **model** would have given rise to at least one better architecture, considering this one was derived loosely from an MNIST architecture I'd developed previously.
- Cleaner Codebase
  - Comments, better code structure (classes and functions), **documentation**, all of it was cut for the sake of time.  This project was lean and mean and honestly is in a state of fairly heavy technical debt as it stands.
- Better AWS Ecosystem usage
  - Using a ssh to start a docker container on an EC2 instance is pretty primitive compared to some of the other services offered (ECS, ECR).  
  - Again, what was familiar was chosen due to time constraints, rather than what is good practice.
- More... committment?
  - Website looks like a meme and a joke (because it is).  The self-constrained budget and time window were a fun gimmick, but definitely emphasizes the point that good software and good infrastructure **takes time**.  
- A better readme/guide
  - A truly helpful guide would have comprehensive step-by-step instructions with pictures.
  
# Instead, here's a rough concept of how to deploy this dumpster fire yourself for whatever terrible reason you have.
  - Use the google_scraper.py or some other method to gather your data
  - Use train.py to train your model until it is "done."  I trained this one on ~1000 examples of each class, with an 85-15 train/test split.
  - Export the model checkpoints to a tensorflow inference serving model with export.py
  - Make/Edit your frontend Flask/Frontend/Website around your inference model.
  - Dockerize your whole project (Flask application including tensorflow and website)
  - Hop on EC2 free tier, snag a free EC2 instance, ssh into it and follow commands_for_ec2.txt
  - Once your EC2 instance is hosting your website, go snag a Domain and point it at your EC2 IP, hopefully with some sort of masking or proxy (optional).
  - That's it, enjoy your own little dumpster fire :)
  


    
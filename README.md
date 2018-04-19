# N-gram Kneser-Ney
An implementation of N-gram language modeling with Kneser-Ney smoothingin Python3. 

# Usage
### Create ngram for training text
```
ngram_m = create_ngram(3,brown.sents(categories="mystery"))
ngram_s = create_ngram(3,brown.sents(categories="science_fiction"))
ngram_r = create_ngram(3,brown.sents(categories="romance"))
```

### Train and fit models for mystery, science fiction and romance 
```
model_m = NgramKN(3,ngram_m)
model_s = NgramKN(3,ngram_s)
model_r = NgramKN(3,ngram_r)
```
### Show model fitting 
```
model_m.fit
model_s.fit
model_r.fit
```

### Classify

```
# A science fiction text from the following link
# https://www.short-story.me/science-fiction-stories/1214-stages-of-grief.html
science_fiction = 'It was hard enough to be forced out of my job, but it was really humiliating to be replaced by a robot. For years robots have been doing repetitive jobs like welding the same spot on products that move down an assembly line. In the last few years they have been doing more sophisticated jobs. They can assemble financial information from the internet and create a first-rate report on the market. They can take patient’s medical history as well as a trained nurse. They can even make diagnoses better than most doctors. The best surgeons now are robots. A human surgeon has to set the thing up, but the robot does the actual cutting, and the result is better than if it had been done by a human doctor.'
```
```
classify(science_fiction,model_s,model_r,model_m)

[OUTPUT]:'science fiction'
```

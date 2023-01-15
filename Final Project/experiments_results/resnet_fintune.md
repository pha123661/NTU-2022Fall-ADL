# Experiment Results for ResNet Fine-tune

## torchvision weight

| bachbone  | Accuracy | training time |
|-----------|----------|---------------|
| ResNet34  | 90.98%   |   2222 sec.   |
| ResNet50  | 91.80%   |   2234 sec.   |
| ResNet101 | 93.44%   |   2290 sec.   |


## clip weight(ViT-B/16)
> training time = train + test one time(about 7sec)

| epoch  | Accuracy | training time |
|--------|----------|---------------|
|   10   | 57.58%   |     1m 15s    |
|   20   | 71.97%   |     2m 17s    |
|   30   | 80.30%   |     3m 16s    |
|   40   | 82.58%   |     4m 19s    |
|   50   | 83.33%   |     5m 18s    |


## clip weight(RN50)
> training time = train + test one time(about 8sec)

| epoch  | Accuracy | training time |
|--------|----------|---------------|
|   10   | 7.58%    | 1m 16s        |
|   20   | 9.09%    | 2m 17s        |
|   30   | 9.85%    | 3m 20s        |
|   40   | 11.36%   | 4m 23s        |
|   50   | 10.61%   | 5m 24s        |

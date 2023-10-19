# Drought Forecasting Using Linear, ConvLSTM and EarthFormer Models. 

<div align="center">
  <img src="./evaluation/gifs/final_github2.gif" width="600" />
</div>


## Experiments results
:sparkles: GIFs [[link]](./evaluation/gifs/)  
:loudspeaker: Presentation [[link]](./output/presentation.pdf)  
:page_with_curl: Report [[link]](./output/report.pdf)  
:mag_right: Different graphics and tables [there](./evaluation/analytics.ipynb)  
:floppy_disk: Latex sourse [[link]](./output/latex.zip)  
:gift: All experiments metrics and configs [[link]](./output/run_summary.csv)  

## Env
```
$ pip install -r env/requirements.txt
```
## Code
Each model in models folder have scripts to generate configs and run it. All experiments can be run with 
```python
(env) python run.py
```

## Data
By default tensor with shape `[T, H, W]`. If you want to add your datasets just add it to the `./data/files` and run `./data//get_config.py`.

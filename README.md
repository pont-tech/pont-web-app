# web-app
This is a web interface for pont.tech cloud frames upscaler and interpolator.

It can be accessed at [https://demo.ponttech.keenetic.pro](https://demo.ponttech.keenetic.pro)
## Demo
![Demo](demo.gif)

## Running locally
First build the container
```
docker built -t pont_tech_web_app .
```
And then run it
```
docker run -p 8501:8501 --gpus=all -d --restart always pont_tech_web_app
```

After that service could be accessed at localhost:8501

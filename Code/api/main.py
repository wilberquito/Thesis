import logging
# import subprocess

import uvicorn

import modular.vision as vision


# def pull_shared_code():
#     subprocess.call(['sh', 'pull_nn_code.sh'])

def main():
#   pull_shared_code()
    logging.basicConfig(filename='myapp.log', level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')
    vision.run()
    uvicorn.run("api:app", host="0.0.0.0", port=8081, reload=True)


if __name__ == '__main__':
    main()

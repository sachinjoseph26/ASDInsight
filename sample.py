import logging as lg
def div(a,b):
    try:
        lg.info(str(a)+"/"+str(b))
        d = a/b
        return d
    except Exception as e:
        print("check log")
        lg.error("division by zero error")
        lg.exception(str(e))
        lg.critical("critical error")

div(4,0)

div(5,0)

div(6,0)

## Testing pipelins

## Fix AB#26

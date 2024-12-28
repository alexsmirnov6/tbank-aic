from dagster import job, op


@op
def simple_op():
    return "Hello, Dagster!"


@job
def my_pipeline():
    simple_op()
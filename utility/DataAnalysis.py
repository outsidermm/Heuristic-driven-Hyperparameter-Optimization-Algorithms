import csv


def write_csv(data: list[dict], header: list[str], filename: str) -> None:
    with open(filename, mode="a+") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writerows(data)


def write_header(header: list[str], filename: str) -> None:
    with open(filename, mode="a+") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()

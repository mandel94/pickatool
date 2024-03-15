# https://stackoverflow.com/questions/37485174/python-locale-in-dateutil-parser
# https://dateutil.readthedocs.io/en/stable/_modules/dateutil/parser/_parser.html#parserinfo
from dateutil import parser
from abc import ABC, abstractmethod
from typing import Literal
from dataclasses import dataclass


class ItalianParserInfo(parser.parserinfo):
    WEEKDAYS = [
        ("Lunedì", "Lun"),
        ("Martedì", "Mar"),
        ("Mercoledì", "Mer"),
        ("Giovedì", "Gio"),
        ("Venerdì", "Ven"),
        ("Sabato", "Sab"),
        ("Domenica", "Dom"),
    ]

    MONTHS = [
        ("Gennaio", "Gen"),
        ("Febbraio", "Feb"),
        ("Marzo", "Mar"),
        ("Aprile", "Apr"),
        ("Maggio", "Mag"),
        ("Giugno", "Giu"),
        ("Luglio", "Lug"),
        ("Agosto", "Ago"),
        ("Settembre", "Set"),
        ("Ottobre", "Ott"),
        ("Novembre", "Nov"),
        ("Dicembre", "Dic"),
    ]

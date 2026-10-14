Real АИСОРИ-M (Web AISORI-M, RIHMI-WDC) exports, trimmed to 14 records per
station, used by `tests/stations/test_rihmi_client.py`. Downloaded 2026-10-14
from http://aisori-m.meteo.ru/waisori/ ("Сутки" section) for stations 22113
(Murmansk) and 22820 (Petrozavodsk):

| file | dataset | delimiter | years |
|---|---|---|---|
| `route_semicolon.zip` | SNMAR, route snow surveys | `;` | 2021–2022 |
| `route_space.zip` | SNMAR, route snow surveys | space (fixed width) | 2020–2024 |
| `snow_cover_semicolon.zip` | SNOW, daily snow cover | `;` | 2023 |
| `snow_cover_space.zip` | SNOW, daily snow cover | space (fixed width) | 2023 |

Each zip is laid out as AISORI-M serves it: `wr<id>.txt` (data, CRLF),
`fld<id>.txt` (column number, width, Russian column name; Windows-1251) and
`statlist<id>.txt` (stations; Windows-1251). meteo.ru describes the data as
openly available («в открытом доступе»).

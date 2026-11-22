import numpy as np


def populate_datetime(record, fails):
    def r(name,value):record("datetime_"+name,value)
    def txt(name,value):r(name,np.frombuffer(value.encode(),dtype=np.uint8))
    for i,(value,unit) in enumerate([
        ("2024","Y"),("2024-03","M"),("1969-12-31","D"),("1969-12-31T23:59:59.987654321","ns"),
        ("2024-02-29T13:14:15.123456","us"),("1970-01-08","W"),
        ("1970-01-01T00:00:00.000001234567","ps"),
        ("1969-12-31T23:59:59.999999999999999999","as"),
        ("2024-02-29","3D"),("2024-02-29T13:14:15","5h"),("NaT","D"),
    ]):
        dt=np.datetime64(value,unit)
        r("parse_"+str(i),dt.astype(np.int64))
        txt("format_"+str(i),np.datetime_as_string(dt))
        u,n=np.datetime_data(dt.dtype);txt("unit_"+str(i),u);r("step_"+str(i),np.array(n,dtype=np.int64))
    for i,(v,unit) in enumerate([(17,"h"),(-12,"3D"),(0,"ms"),(-9223372036854775808,"ns")]):
        r("timedelta_"+str(i),np.timedelta64(v,unit).astype(np.int64))
        r("timedelta_nat_"+str(i),np.array(np.isnat(np.timedelta64(v,unit))))
    for i,(v,unit) in enumerate([(0,"s"),(-7,"D"),(32768,"ms"),(-9223372036854775808,"us")]):
        dt=np.datetime64(v,unit)
        r("scalar_ticks_"+str(i),dt.astype(np.int64));txt("scalar_format_"+str(i),np.datetime_as_string(dt))
    vals=np.array(["2024-01-05","2024-01-06","NaT","1969-12-31"],dtype="datetime64[D]").reshape(2,2)
    r("array_ticks",vals.astype(np.int64));r("isnat",np.isnat(vals))
    r("cast_hours",vals.astype("datetime64[h]").astype(np.int64))
    r("cast_months",vals.astype("datetime64[M]").astype(np.int64))
    calendar=np.busdaycalendar(weekmask="Mon Tue Wed Thu Fri",holidays=["2024-01-01","2024-01-06","2024-01-01","2024-01-15","NaT"])
    r("calendar_weekmask",calendar.weekmask);r("calendar_holidays",calendar.holidays.astype(np.int64))
    dates=np.array(["2024-01-01","2024-01-05","2024-01-06","2024-01-08","2024-01-15","NaT"],dtype="datetime64[D]")
    r("days",dates.astype(np.int64));r("is_busday",np.is_busday(dates,busdaycal=calendar))
    begin=np.array([["2024-01-05"],["2024-01-08"],["2024-01-15"]],dtype="datetime64[D]")
    end=np.array(["2024-01-06","2024-01-19","2023-12-29"],dtype="datetime64[D]")
    r("begin",begin.astype(np.int64));r("end",end.astype(np.int64))
    r("busday_count",np.busday_count(begin,end,busdaycal=calendar))
    starts=np.array(["2024-01-06","2024-01-31","2024-03-31","2024-01-15"],dtype="datetime64[D]").reshape(-1,1)
    offsets=np.array([-7,0,5,42],dtype=np.int64)
    r("starts",starts.astype(np.int64));r("offsets",offsets)
    for roll in ("forward","backward","following","preceding","modifiedfollowing","modifiedpreceding","nat"):
        r("busday_offset_"+roll,np.busday_offset(starts,offsets,roll=roll,busdaycal=calendar).astype(np.int64))
    other=np.busdaycalendar(weekmask="Mon Wed Sat",holidays=["2024-01-06","2024-01-10"])
    r("custom_count",np.busday_count(begin,end,busdaycal=other))
    r("custom_offset",np.busday_offset(starts,offsets,roll="forward",busdaycal=other).astype(np.int64))
    fails("datetime_bad_date",lambda:np.datetime64("2023-02-29","D"))
    fails("datetime_bad_unit",lambda:np.datetime64(1,"invalid"))
    fails("datetime_busday_count_nat",lambda:np.busday_count(np.datetime64("NaT","D"),np.datetime64("2024-01-01","D")))
    fails("datetime_busday_offset_raise",lambda:np.busday_offset("2024-01-07",1))
    fails("datetime_calendar_empty_week",lambda:np.busdaycalendar(weekmask="0000000"))

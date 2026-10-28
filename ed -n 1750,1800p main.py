[1mdiff --git a/main.py b/main.py[m
[1mindex 85dd9ed..77b87fa 100644[m
[1m--- a/main.py[m
[1m+++ b/main.py[m
[36m@@ -1763,13 +1763,26 @@[m [masync def staff_access_guard(request: Request, call_next):[m
             )[m
         return RedirectResponse(url=login_path, status_code=303)[m
 [m
[31m-    if staff.role != required_role:[m
[32m+[m[41m    [m
[32m+[m[32m    staff_role = str(getattr(staff, "role", "") or "").strip().upper()[m[41m[m
[32m+[m[32m    expected_role = str(required_role or "").strip().upper()[m[41m[m
[32m+[m[41m[m
[32m+[m[32m    if staff_role != expected_role:[m[41m[m
         if path.startswith("/api/"):[m
             return JSONResponse([m
                 status_code=403,[m
[31m-                content={"status": "error", "detail": "Insufficient staff role"}[m
[32m+[m[32m                content={[m[41m[m
[32m+[m[32m                    "status": "error",[m[41m[m
[32m+[m[32m                    "detail": "Insufficient staff role",[m[41m[m
[32m+[m[32m                    "expected_role": expected_role,[m[41m[m
[32m+[m[32m                    "actual_role": staff_role[m[41m[m
[32m+[m[32m                }[m[41m[m
             )[m
[31m-        return RedirectResponse(url=f"{login_path}?role_error=1", status_code=303)[m
[32m+[m[32m        return RedirectResponse([m[41m[m
[32m+[m[32m            url=f"{login_path}?role_error=1",[m[41m[m
[32m+[m[32m            status_code=303[m[41m[m
[32m+[m[32m        )[m[41m[m
[32m+[m[41m[m
 [m
     request.state.staff_user = staff[m
     return await call_next(request)[m

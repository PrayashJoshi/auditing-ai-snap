import sys, re
from pypdf import PdfReader
r = PdfReader(sys.argv[1])
txt = "\n".join((p.extract_text() or "") + f"\n=== [page {i+1}] ===" for i,p in enumerate(r.pages))
open(sys.argv[2],'w').write(txt)
fields = r.get_fields() or {}
print("pages:", len(r.pages), " fillable fields:", len(fields), " words:", len(txt.split()))
print("revision strings:", sorted(set(re.findall(r'Rev\.\s*\d{1,2}-\d{2}', txt)))[:6])
for kw in ['Food Assistance','SNAP','authorized representative','Authorized Representative','heat','utility','shelter','rent','earned','child support','dependent care','medical','vehicle','bank','citizen','Social Security']:
    print(f"{kw!r}: {len(re.findall(re.escape(kw), txt, flags=re.I))}", end='; ')
print()

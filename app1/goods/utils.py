from django.db.models import Q
from django.contrib.postgres.search import SearchVector, SearchQuery, SearchRank, SearchHeadline

from goods.models import Products

def q_search(query):
    
    if query.isdigit() and len(query) <= 5:
        return Products.objects.filter(id=int(query))
    
    vector = SearchVector("name", "description", "quantity")
    search_query = SearchQuery(query)

    results = (
        Products.objects.annotate(
            rank=SearchRank(vector, search_query)
        )
        .filter(rank__gt=0) 
        .order_by("-rank") 
    )

    results = results.annotate(
        headline=SearchHeadline(
            "name", search_query, start_sel='<mark style="background-color: yellow;">', stop_sel="</mark>"
        )
    )
    results = results.annotate(
        bodyline=SearchHeadline(
            "description", search_query, start_sel='<mark style="background-color: yellow;">', stop_sel="</mark>"
        )
    )
    results = results.annotate(
        location=SearchHeadline(
            "quantity", search_query, start_sel='<mark style="background-color: yellow;">', stop_sel="</mark>"
        )
    )

    return results
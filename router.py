def route_question(question):

    q = question.lower()
    routes = []
    if any(word in q for word in ["market", "trend", "industry", "growth"]):
        routes.append("industry")
    if any(word in q for word in ["competitor", "company", "microsoft", "nvidia", "tesla"]):
        routes.append("companies")
    if any(word in q for word in ["latest", "recent", "news", "today"]):
        routes.append("news")
    if len(routes) == 0:
        routes = ["companies", "industry"]
    return routes
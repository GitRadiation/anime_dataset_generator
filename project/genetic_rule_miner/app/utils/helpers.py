def update_controls(query, full_series_list, checkbox_map, search_bar_ref):
    lower_query = query.lower()
    filtered = [s for s in full_series_list if lower_query in s.lower()][:10]
    search_bar_ref.current.controls = [checkbox_map[s] for s in filtered]
    search_bar_ref.current.update()

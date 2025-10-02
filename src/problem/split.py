def split_tour_to_routes(tour, instance):

    capacity = instance.capacity
    get_demand = lambda customer: instance.customers[customer]['demand']


    routes = []
    current_route = [0]
    current_load = 0

    for customer in tour:
        demand = get_demand(customer)
        if current_load + demand <= capacity:
            current_route.append(customer)
            current_load += demand
        else:
            current_route.append(0)
            routes.append(current_route)
            # Starts a new route with given customer
            current_route = [0, customer]
            current_load = demand

    current_route.append(0)
    routes.append(current_route)
    return routes

def routes_to_tour(routes):
    tour = []
    for route in routes:
        for customer in route:
            if customer != 0:
                tour.append(customer)
    return tour

def tour_to_routes(tour, instance):

    return split_tour_to_routes(tour, instance)
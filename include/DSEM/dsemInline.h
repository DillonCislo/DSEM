/*
 * Copyright (C) 2020 Dillon Cislo
 *
 * This file is part of DSEN++.
 *
 * DSEN++ is free software: you can redistribute it and/or modify it under the terms
 * of the GNU General Public License as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will by useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program.
 * If not, see <http://www.gnu.org/licenses/>
 *
 */

#ifdef DSEM_INLINE
#undef DSEM_INLINE
#endif

#ifndef DSEM_STATIC_LIBRARY
#  define DSEM_INLINE inline
#else
#  define DSEM_INLINE
#endif
